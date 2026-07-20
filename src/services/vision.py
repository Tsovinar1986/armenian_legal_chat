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

# detect_objects() is restricted to YOLO COCO classes [0, 67] = person, cell
# phone — so "cell phone" is currently the only non-person class that can
# ever show up here. Extend this if that class list ever grows.
OBJECT_NAME_HY = {
    "cell phone": "հեռախոս",
}

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

    def _draw_unicode_text(self, frame, text, position, font_size=20, bg=False):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        if bg:
            pad = 6
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                fill=(0, 0, 0),
            )
        draw.text(position, text, font=font, fill=(0, 255, 0))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def process_video(self, video_path, window_name="Legal AI - Video Analysis", sample_every=5, max_width=960):
        """Windowed playback + analysis, tuned to stay usable on an 8GB-RAM
        machine: uploaded videos are often shot at 4K, and running YOLO+
        MediaPipe at full resolution on every single frame is the actual
        RAM/CPU cost — not the window itself. Two levers:
          - every frame is downscaled to max_width before both detection and
            display (a 3840-wide frame costs far more to run through the
            model than a 960-wide one, and the display window doesn't need
            more than that to be watchable);
          - the heavy detection pipeline only runs every `sample_every`
            frames; frames in between just replay the last known status
            text, so playback stays smooth without re-analyzing every frame.
        The status bar uses a larger font with a solid background bar (see
        _draw_unicode_text's bg=True) so it stays readable over busy video
        content instead of blending into whatever's behind it. Press 'q' to
        stop early.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        unique_actions = set()
        last_emotion = None
        last_objects = []
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / w
                    frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

                if frame_idx % sample_every == 0:
                    frame = self.process_frame(frame)
                    current_actions = self.state.get_actions()
                    if current_actions:
                        unique_actions.update(current_actions)
                    last_emotion = self.state.get_emotion() or last_emotion
                    # Object list is per-frame, not accumulated like actions —
                    # it only ever holds something when the LAST analyzed frame
                    # had no person in it (see process_frame), so an empty list
                    # here means either nothing at all, or a person is present
                    # (in which case unique_actions/last_emotion tell the story).
                    last_objects = self.state.get_objects()

                # Three distinct states, not two: a person present (actions +
                # emotion), an object but no person (object detection only —
                # no fabricated action/emotion for something that isn't a
                # person), or nothing detected yet.
                if unique_actions or last_emotion:
                    status_text = (
                        ("Հայտնաբերված գործողություններ. " + ", ".join(sorted(unique_actions))
                         if unique_actions else "Անձ է հայտնաբերվել")
                        + f" | Էմոցիան. {last_emotion or '...'}"
                    )
                elif last_objects:
                    status_text = "Օբյեկտների հայտնաբերում (անձ չի հայտնաբերվել). " + ", ".join(last_objects)
                else:
                    status_text = "Վերլուծություն..."
                frame = self._draw_unicode_text(frame, status_text, (10, 10), font_size=24, bg=True)

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1
        finally:
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
        person_present = False

        for r in results:
            for box in r.boxes:
                if self.classifier.yolo.names[int(box.cls[0])] == 'person':
                    person_present = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    actions = self.classifier.classify_actions(crop, objects_seen)
                    actions_in_frame.extend(actions)
                    for idx, action in enumerate(actions):
                        frame = self._draw_unicode_text(frame, action, (x1, y1 - 30 - idx * 22))

        # Non-person objects (e.g. a phone) seen in a frame with nobody in it —
        # tracked separately so callers can say "object detection only" instead
        # of defaulting to a fake action/emotion for content with no person.
        non_person_objects = sorted({
            OBJECT_NAME_HY.get(name, name) for name in objects_seen if name != 'person'
        })
        self.state.update_objects(non_person_objects if not person_present else [])

        # Only read an emotion off a frame that actually has a person in it —
        # detect_emotion() still runs its own face search either way, but
        # there's no point asking "what's the emotion" of an empty frame.
        emotion = self.classifier.detect_emotion(frame) if person_present else None
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
