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

# Non-person classes VisionClassifier.detect_objects can return (see its
# _DETECTED_CLASSES). Keyed per interface language ("hy"/"en"/"ru"), same
# convention as VisionClassifier.action_map/emotion_map — falls back to "en"
# via _object_names() below for any other language code.
OBJECT_NAME_MAP = {
    "hy": {
        "cell phone": "հեռախոս",
        "bicycle": "հեծանիվ",
        "car": "մեքենա",
        "motorcycle": "մոտոցիկլ",
        "bus": "ավտոբուս",
        "truck": "բեռնատար",
    },
    "en": {
        "cell phone": "phone",
        "bicycle": "bicycle",
        "car": "car",
        "motorcycle": "motorcycle",
        "bus": "bus",
        "truck": "truck",
    },
    "ru": {
        "cell phone": "телефон",
        "bicycle": "велосипед",
        "car": "автомобиль",
        "motorcycle": "мотоцикл",
        "bus": "автобус",
        "truck": "грузовик",
    },
}

# All other hardcoded UI strings this service draws on frames / builds status
# text from, keyed the same way.
UI_TEXT = {
    "hy": {
        "emotion_label": "Էմոցիան",
        "therapist_suggestion": "💙 Առաջարկվում է խոսել թերապևտի հետ",
        "objects_no_person": "Օբյեկտների հայտնաբերում (անձ չի հայտնաբերվել)",
        "latest_detection": "Վերջին հայտնաբերումը",
        "analyzing": "Վերլուծություն...",
    },
    "en": {
        "emotion_label": "Emotion",
        "therapist_suggestion": "💙 Talking to a therapist is recommended",
        "objects_no_person": "Object detected (no person found)",
        "latest_detection": "Latest detection",
        "analyzing": "Analyzing...",
    },
    "ru": {
        "emotion_label": "Эмоция",
        "therapist_suggestion": "💙 Рекомендуется поговорить с терапевтом",
        "objects_no_person": "Обнаружены объекты (человек не обнаружен)",
        "latest_detection": "Последнее обнаружение",
        "analyzing": "Анализ...",
    },
}


def _ui_text(language):
    return UI_TEXT.get(language) or UI_TEXT["en"]


def _object_names(language):
    return OBJECT_NAME_MAP.get(language) or OBJECT_NAME_MAP["en"]

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


MENTAL_HEALTH_SUGGESTION = {
    "hy": (
        "💙 Վերջին րոպեների ընթացքում նկատվում է տխուր/անհանգիստ տրամադրություն։ "
        "Սա միայն դեմքի արտահայտության վրա հիմնված մոտավոր դիտարկում է, ոչ թե "
        "ախտորոշում։ Եթե դա իրական է Ձեզ համար, խորհուրդ ենք տալիս զրուցել "
        "որակավորված թերապևտի հետ կամ գրել մեր չաթում աջակցության համար։"
    ),
    "en": (
        "💙 A sad/anxious mood has been noticed over the last few minutes. "
        "This is only a rough observation based on facial expression, not a "
        "diagnosis. If this feels real for you, we recommend talking to a "
        "qualified therapist or writing to us in chat for support."
    ),
    "ru": (
        "💙 За последние несколько минут замечено грустное/тревожное "
        "настроение. Это лишь приблизительное наблюдение на основе выражения "
        "лица, а не диагноз. Если это актуально для вас, рекомендуем "
        "поговорить с квалифицированным терапевтом или написать нам в чат за "
        "поддержкой."
    ),
}


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

    def _check_mental_health_concern(self, emotion, negative_labels, language='hy'):
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
            suggestion = MENTAL_HEALTH_SUGGESTION.get(language) or MENTAL_HEALTH_SUGGESTION["en"]
            self.state.update_mental_health_concern(True, suggestion)
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

    def process_video(self, video_path, window_name="Legal AI - Video Analysis", sample_every=5, max_width=960, language='hy'):
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
                    frame = self.process_frame(frame, language=language)  # draws its own per-person action + emotion labels
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
                    ui = _ui_text(language)
                    if last_objects:
                        status_text = f"{ui['objects_no_person']}: " + ", ".join(last_objects)
                    elif unique_actions or last_emotion:
                        status_text = f"{ui['latest_detection']}: {', '.join(sorted(unique_actions)) or last_emotion}"
                    else:
                        status_text = ui['analyzing']
                    frame = self._draw_unicode_text(frame, status_text, (10, 10), font_size=24, bg=True)

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_idx += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return list(unique_actions)

    def analyze_video_headless(self, video_path, max_frames=12, language='hy'):
        """Same detection pipeline as process_video, but for server use: no
        cv2.imshow/waitKey (those need a display and would hang/crash a
        request handler), and it samples up to max_frames evenly across the
        video instead of every frame, since a full-resolution upload run
        frame-by-frame inside an HTTP request would be far too slow.
        Deliberately doesn't touch self.state — that's shared across every
        request this process handles, and mixing concurrent uploads into one
        SystemState would corrupt all of them.

        `language` ("hy"/"en"/"ru") selects which language the returned
        action/emotion labels are written in, matching the app's interface
        language for this request rather than always Armenian."""
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
        object_samples = []
        object_names = _object_names(language)
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
                    # get_person_boxes de-dupes overlapping/nested boxes YOLO's
                    # own NMS misses — see its docstring for why iterating
                    # r.boxes directly here double-counted one real person's
                    # pose as several contradictory "people".
                    for x1, y1, x2, y2 in self.classifier.get_person_boxes(results):
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        frame_actions.extend(self.classifier.classify_actions(crop, objects_seen, language=language))
                    action_samples.append(frame_actions)
                    # Emotion detection disabled for now (commented out, not
                    # removed — re-enable by uncommenting this call plus the
                    # emotion_changes block and the "emotion"/"emotion_changes"
                    # entries in the return dict below).
                    # detect_emotion() runs its own independent face search
                    # (MediaPipe FaceMesh / Haar cascade) and already returns
                    # None on its own when no face is found — gating this on
                    # YOLO's separate person-box detection too was stricter
                    # than necessary and could suppress a real, visible face
                    # (e.g. a close-up shot) that YOLO's person detector
                    # didn't confidently box.
                    # emotion_samples.append(self.classifier.detect_emotion(frame, language=language))
                    # Non-person objects (vehicles, a phone, etc. — see
                    # VisionClassifier._DETECTED_CLASSES) tracked regardless of
                    # whether a person is also in the frame, same as
                    # process_frame — this pipeline previously didn't track
                    # objects at all, so a car/bicycle in an uploaded video
                    # never showed up in the returned result.
                    object_samples.append({
                        object_names.get(name, name) for name in objects_seen if name != 'person'
                    })
                    frames_analyzed += 1
                frame_idx += 1
        finally:
            cap.release()

        # Emotion detection disabled for now (see emotion_samples.append(...)
        # above, commented out) — emotion_changes computation and the
        # "emotion"/"emotion_changes" return entries are commented out to
        # match, not removed. To re-enable: uncomment all three.
        # Distinct emotions in the order they were first seen, collapsing
        # consecutive repeats — e.g. [neutral, sad] means the face's
        # expression (mouth shape is the main signal _infer_emotion_from_landmarks
        # reads) shifted from neutral to sad at some point in the video, not
        # just what the majority-vote final emotion happened to be.
        # emotion_changes = []
        # for e in emotion_samples:
        #     if e and (not emotion_changes or emotion_changes[-1] != e):
        #         emotion_changes.append(e)

        # min_ratio=0.25 (needing recurrence in only ~3 of 12 sampled frames)
        # was measured against real footage to still let contradictory noise
        # through as "confirmed" fact — e.g. "running" and "pointing" each
        # firing on a couple of noisy frames of someone just sitting/standing
        # talking to the camera, alongside plausible ones like "phone use".
        # 0.4 (matching process_video's own default) requires a real majority
        # of sampled frames to agree before an action is reported.
        return {
            "actions": sorted(_confirmed_items(action_samples, min_ratio=0.4)),
            "emotion": None,
            "emotion_changes": [],
            "objects": sorted(_confirmed_items(object_samples, min_ratio=0.4)),
            "frames_analyzed": frames_analyzed,
        }

    def process_frame(self, frame, language='hy'):
        ui = _ui_text(language)
        results, objects_seen = self.classifier.detect_objects(frame)
        actions_in_frame = []

        # get_person_boxes de-dupes overlapping/nested boxes YOLO's own NMS
        # misses — see its docstring for why iterating r.boxes directly here
        # double-counted one real person's pose as several contradictory
        # "people".
        for x1, y1, x2, y2 in self.classifier.get_person_boxes(results):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            actions = self.classifier.classify_actions(crop, objects_seen, language=language)
            actions_in_frame.extend(actions)
            for idx, action in enumerate(actions):
                frame = self._draw_unicode_text(frame, action, (x1, y1 - 30 - idx * 22))

        # Non-person objects (a phone, a car/bicycle/etc.) — tracked
        # regardless of whether a person is also in the frame. Previously
        # this was zeroed out whenever person_present was True, which
        # silently dropped exactly the common case of "a person next to a
        # vehicle" (e.g. a traffic stop, someone working on a car) instead
        # of reporting both.
        object_names = _object_names(language)
        non_person_objects = sorted({
            object_names.get(name, name) for name in objects_seen if name != 'person'
        })
        self.state.update_objects(non_person_objects)

        # detect_emotion() runs its own independent face search (MediaPipe
        # FaceMesh / Haar cascade) and already returns None on its own when
        # no face is found — gating this on YOLO's separate person-box
        # detection too (as an earlier version of this method did) was
        # stricter than necessary and could suppress a real, visible face
        # (e.g. a close-up shot) that YOLO's person detector didn't
        # confidently box, making emotion silently stop showing.
        emotion = self.classifier.detect_emotion(frame, language=language)
        self.state.update_emotion(emotion)
        if emotion:
            frame = self._draw_unicode_text(frame, f"{ui['emotion_label']}: {emotion}", (10, 70))

        emotion_labels = self.classifier._emotion_labels(language)
        negative_labels = {
            emotion_labels.get('sad'),
            emotion_labels.get('angry'),
            # Lip-biting reads as a nervous/anxious tell, the same "anxious
            # mood" MENTAL_HEALTH_SUGGESTION already talks about — not just
            # sad/angry expressions.
            emotion_labels.get('biting_lip'),
        }
        self._check_mental_health_concern(emotion, negative_labels, language=language)
        if self.state.get_mental_health_concern():
            frame = self._draw_unicode_text(frame, ui['therapist_suggestion'], (10, 100))

        self.state.update_actions(list(set(actions_in_frame)))
        return frame


if __name__ == "__main__":
    print("LegalVisionService module loaded.")
