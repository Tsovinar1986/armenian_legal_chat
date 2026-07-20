from collections import Counter, deque

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

def _confirmed_items(recent_samples, min_ratio=0.4):
    """recent_samples: an iterable of per-sampled-frame collections (action
    lists, one per recent classify_actions() call). Returns only the items
    that recurred in at least min_ratio of those samples.

    Why this exists: classify_actions() runs independently on every sampled
    frame with no memory of previous frames, so a single noisy pose estimate
    (motion blur, a bad crop, an unlucky landmark jitter) can spuriously fire
    an action that never really happened — and since callers were adding
    every sampled frame's raw output straight into the reported set, that
    one-off noise became a permanent, confidently-reported "detection".
    Requiring recurrence across a small rolling window (see the deque this is
    called with) filters that out while still catching genuinely sustained
    behavior within roughly a second of real time.
    """
    samples = list(recent_samples)
    if not samples:
        return set()
    counts = Counter()
    for sample in samples:
        counts.update(set(sample))
    threshold = max(1, round(len(samples) * min_ratio))
    return {item for item, n in counts.items() if n >= threshold}


def _majority_emotion(recent_emotion_samples):
    """Same idea as _confirmed_items but for the single current emotion
    value: majority vote across the recent window instead of trusting
    whatever the single latest sample happened to say, which is what made
    the displayed emotion flicker between unrelated labels every sample."""
    non_none = [e for e in recent_emotion_samples if e]
    if not non_none:
        return None
    return Counter(non_none).most_common(1)[0][0]


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
        Emotion and per-person action labels are drawn directly on the frame
        by process_frame itself (right next to the person they're about) —
        that's the primary "seen in the video" display. This method only
        adds a top-left status bar (larger font, solid background so it
        stays readable over busy video content) for what process_frame
        doesn't already cover: a frame with an object but no person, or
        nothing detected yet. The terminal's final summary (see
        handle_upload in src/main.py) is unchanged — this is purely about
        what's visible in the frame while it plays. Press 'q' to stop early.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        unique_actions = set()
        last_emotion = None
        last_objects = []
        # Rolling window over the last few *sampled* frames (not raw video
        # frames) — an action/emotion only gets folded into unique_actions/
        # last_emotion (and therefore into the final report and the fallback
        # status bar) once it recurs across this window. See _confirmed_items'
        # docstring for why: a single sampled frame's raw classify_actions()
        # output is not reliable enough to report as fact on its own.
        recent_actions = deque(maxlen=5)
        recent_emotions = deque(maxlen=5)
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

                person_in_frame = False
                if frame_idx % sample_every == 0:
                    frame = self.process_frame(frame)  # draws its own per-person action + emotion labels
                    current_actions = self.state.get_actions()
                    current_emotion = self.state.get_emotion()
                    person_in_frame = bool(current_actions or current_emotion)

                    recent_actions.append(current_actions)
                    unique_actions.update(_confirmed_items(recent_actions))
                    recent_emotions.append(current_emotion)
                    last_emotion = _majority_emotion(recent_emotions) or last_emotion

                    last_objects = self.state.get_objects() if not person_in_frame else []

                # Only add the aggregate bar for what process_frame's own
                # overlay doesn't already show for THIS frame: an object with
                # no person (process_frame draws nothing in that case), or
                # nothing detected yet. When a person is in frame, its own
                # per-person action/emotion labels are the display — no need
                # to duplicate that in a second, less specific line.
                if not person_in_frame:
                    if last_objects:
                        status_text = "Օբյեկտների հայտնաբերում (անձ չի հայտնաբերվել). " + ", ".join(last_objects)
                    elif unique_actions or last_emotion:
                        status_text = f"Վերջին հայտնաբերումը. {', '.join(sorted(unique_actions)) or last_emotion}"
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

        # Collected across ALL sampled frames, then confirmed/majority-voted
        # once at the end (see _confirmed_items/_majority_emotion) instead of
        # trusting each sampled frame's raw output as fact on its own — a
        # single noisy pose estimate from one of the (sparse, evenly-spaced)
        # sampled frames shouldn't become a permanent line in the report.
        action_samples = []
        emotion_samples = []
        frames_analyzed = 0
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if sample_indices is None or frame_idx in sample_indices:
                    results, objects_seen = self.classifier.detect_objects(frame)
                    frame_actions = []
                    for r in results:
                        for box in r.boxes:
                            if self.classifier.yolo.names[int(box.cls[0])] == 'person':
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                crop = frame[y1:y2, x1:x2]
                                if crop.size == 0:
                                    continue
                                frame_actions.extend(self.classifier.classify_actions(crop, objects_seen))
                    action_samples.append(frame_actions)
                    # detect_emotion() runs its own independent face search
                    # (MediaPipe FaceMesh / Haar cascade) and already returns
                    # None on its own when no face is found — gating this on
                    # YOLO's separate person-box detection too was stricter
                    # than necessary and could suppress a real, visible face
                    # (e.g. a close-up shot) that YOLO's person detector
                    # didn't confidently box.
                    emotion_samples.append(self.classifier.detect_emotion(frame))
                    frames_analyzed += 1
                frame_idx += 1
        finally:
            cap.release()

        return {
            "actions": sorted(_confirmed_items(action_samples, min_ratio=0.25)),
            "emotion": _majority_emotion(emotion_samples),
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

        # detect_emotion() runs its own independent face search (MediaPipe
        # FaceMesh / Haar cascade) and already returns None on its own when
        # no face is found — gating this on YOLO's separate person-box
        # detection too (as an earlier version of this method did) was
        # stricter than necessary and could suppress a real, visible face
        # (e.g. a close-up shot) that YOLO's person detector didn't
        # confidently box, making emotion silently stop showing.
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
