from collections import deque

import cv2
import numpy as np
import unicodedata
from PIL import Image, ImageDraw, ImageFont

# How many recent frames' emotions to consider, and what fraction of them must be
# negative (sad/angry) before surfacing a therapist suggestion. This is a coarse,
# heuristic UX nudge based on facial-expression heuristics in VisionClassifier — it
# is NOT a clinical assessment and must never be presented as one. It only ever
# suggests talking to a therapist; it never claims to detect a diagnosis.
MENTAL_HEALTH_CONCERN_WINDOW = 30
MENTAL_HEALTH_CONCERN_RATIO = 0.6

MENTAL_HEALTH_SUGGESTION_HY = (
    "💙 Վերջին րոպեների ընթացքում նկատվում է տխուր/անհանգիստ տրամադրություն։ "
    "Սա միայն դեմքի արտահայտության վրա հիմնված մոտավոր դիտարկում է, ոչ թե "
    "ախտորոշում։ Եթե դա իրական է Ձեզ համար, խորհուրդ ենք տալիս զրուցել "
    "որակավորված թերապևտի հետ կամ գրել մեր չաթում աջակցության համար։"
)


class LegalVisionService:
    def __init__(self, state):
        self.state = state
        self._classifier = None
        self.font_path = "/Library/Fonts/Arial Unicode.ttf"
        self._recent_emotions = deque(maxlen=MENTAL_HEALTH_CONCERN_WINDOW)

    @property
    def classifier(self):
        """Lazily import and build VisionClassifier (PyTorch + YOLO + MediaPipe) only
        on first actual use (webcam frame or video upload), so text/voice-only sessions
        never pay that memory/import cost."""
        if self._classifier is None:
            print("🎥 Loading vision models (YOLO + MediaPipe) on first use...")
            from src.services.vision_classifier import VisionClassifier
            self._classifier = VisionClassifier()
        return self._classifier

    def _check_mental_health_concern(self, emotion, negative_labels):
        """Track a rolling window of recent per-frame emotions and flag a soft
        therapist suggestion once a sustained-enough share are negative.

        A single sad/angry frame is noisy (lighting, a blink, a frown mid-sentence)
        so this only fires on a sustained pattern across MENTAL_HEALTH_CONCERN_WINDOW
        frames, and clears itself once the pattern is no longer sustained.
        """
        if not emotion:
            return
        self._recent_emotions.append(emotion)
        negative_count = sum(1 for e in self._recent_emotions if e in negative_labels)
        ratio = negative_count / len(self._recent_emotions)
        is_full_window = len(self._recent_emotions) == self._recent_emotions.maxlen
        if is_full_window and ratio >= MENTAL_HEALTH_CONCERN_RATIO:
            self.state.update_mental_health_concern(True, MENTAL_HEALTH_SUGGESTION_HY)
        else:
            self.state.update_mental_health_concern(False)

    def _draw_unicode_text(self, frame, text, position):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(self.font_path, 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=(0, 255, 0))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def process_video(self, video_path, window_name="Legal AI - Video Analysis"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        unique_actions = set()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            current_actions = self.state.get_actions()
            if current_actions:
                unique_actions.update(current_actions)

            current_emotion = self.state.get_emotion()
            if current_emotion:
                processed_frame = self._draw_unicode_text(processed_frame, f"Էմոցիան: {current_emotion}", (10, 70))

            status_text = (
                "Detected: " + ", ".join(sorted(unique_actions)) + f" | Էմոցիան: {current_emotion}"
                if unique_actions else f"Detecting actions... | Էմոցիան: {current_emotion}"
            )
            # cv2.putText's built-in Hershey font is ASCII/Latin-only — Armenian
            # characters in status_text rendered as "?????" boxes. _draw_unicode_text
            # (PIL + a Unicode-capable font) already handles this correctly for the
            # other overlays below; this call was just left on the broken path.
            processed_frame = self._draw_unicode_text(processed_frame, status_text, (10, 40))

            cv2.imshow(window_name, processed_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return list(unique_actions)

    def analyze_video_headless(self, video_path, max_frames=12):
        """Same detection pipeline as process_video, but for server use: no
        cv2.imshow/waitKey (those need a display and would hang/crash a
        request handler), and it samples up to max_frames evenly across the
        video instead of every frame, since a full-resolution upload run
        frame-by-frame inside an HTTP request would be far too slow.
        Deliberately doesn't touch self.state — that's shared across every
        request this process handles, and mixing concurrent uploads into one
        SystemState would corrupt all of them."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        sample_indices = (
            {round(i * total_frames / max_frames) for i in range(max_frames)}
            if total_frames > max_frames else None
        )

        unique_actions = set()
        last_emotion = None
        frames_analyzed = 0
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if sample_indices is None or frame_idx in sample_indices:
                    results, objects_seen = self.classifier.detect_objects(frame)
                    for r in results:
                        for box in r.boxes:
                            if self.classifier.yolo.names[int(box.cls[0])] == 'person':
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                crop = frame[y1:y2, x1:x2]
                                if crop.size == 0:
                                    continue
                                unique_actions.update(self.classifier.classify_actions(crop, objects_seen))
                    last_emotion = self.classifier.detect_emotion(frame) or last_emotion
                    frames_analyzed += 1
                frame_idx += 1
        finally:
            cap.release()

        return {
            "actions": sorted(unique_actions),
            "emotion": last_emotion,
            "frames_analyzed": frames_analyzed,
        }

    def process_frame(self, frame):
        results, objects_seen = self.classifier.detect_objects(frame)
        actions_in_frame = []

        for r in results:
            for box in r.boxes:
                if self.classifier.yolo.names[int(box.cls[0])] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    actions = self.classifier.classify_actions(crop, objects_seen)
                    actions_in_frame.extend(actions)
                    for idx, action in enumerate(actions):
                        frame = self._draw_unicode_text(frame, action, (x1, y1 - 30 - idx * 22))

        emotion = self.classifier.detect_emotion(frame)
        self.state.update_emotion(emotion)
        if emotion:
            frame = self._draw_unicode_text(frame, f"Էմոցիան: {emotion}", (10, 70))

        negative_labels = {
            self.classifier.emotion_map.get('sad'),
            self.classifier.emotion_map.get('angry'),
        }
        self._check_mental_health_concern(emotion, negative_labels)
        if self.state.get_mental_health_concern():
            frame = self._draw_unicode_text(frame, "💙 Առաջարկվում է խոսել թերապևտի հետ", (10, 100))

        self.state.update_actions(list(set(actions_in_frame)))
        return frame


if __name__ == "__main__":
    print("LegalVisionService module loaded.")
