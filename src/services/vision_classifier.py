import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
import numpy as np


class VisionClassifier:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt') if YOLO is not None else None
        self.mp_pose = None
        self.mp_face = None
        if mp is not None:
            self.mp_pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            try:
                self.mp_face = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
            except Exception:
                self.mp_face = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Only genuinely legal-relevant actions carry a parenthetical legal
        # gloss (kept so those still get explicitly "called out" for case
        # review). Everyday actions (walking, standing, sitting, etc.) are
        # reported plainly — forcing every mundane action through a legal
        # interpretation (e.g. "standing = sign of respect") misrepresented
        # ordinary footage that has nothing to do with a legal incident.
        #
        # Labels are keyed per interface language ("hy"/"en"/"ru") so the
        # displayed action/emotion text matches whatever language the app's
        # UI is running in, instead of always being Armenian. classify_actions
        # and detect_emotion take a `language` argument and fall back to "en"
        # for any language code without its own table (mirrors legal_agent.py's
        # _t() convention).
        self.action_map = {
            'hy': {
                'hand_up': 'Ձեռքի բարձրացում (Խոսքի իրավունքի խնդրանք)',
                'hands_on_hips': 'Ձեռքեր գոտկատեղին (Պաշտպանողական դիրք)',
                'pointing': 'Ցույց տալ (Ուղղորդող նշում)',
                'bent_over': 'Ծռված դիրք (Ուշադրության կենտրոնացում)',
                'walking': 'Քայլում է',
                'running': 'Վազում է',
                'phone': 'Հեռախոսի օգտագործում',
                'standing': 'Կանգնած',
                'sitting': 'Նստած',
                'arms_crossed': 'Ձեռքերը խաչած',
                'normal': 'Բնական վիճակ',
            },
            'en': {
                'hand_up': 'Hand raised (request to speak)',
                'hands_on_hips': 'Hands on hips (defensive posture)',
                'pointing': 'Pointing (directive gesture)',
                'bent_over': 'Bent over (focused attention)',
                'walking': 'Walking',
                'running': 'Running',
                'phone': 'Using phone',
                'standing': 'Standing',
                'sitting': 'Sitting',
                'arms_crossed': 'Arms crossed',
                'normal': 'Natural state',
            },
            'ru': {
                'hand_up': 'Поднятая рука (просьба слова)',
                'hands_on_hips': 'Руки на поясе (защитная поза)',
                'pointing': 'Указывающий жест',
                'bent_over': 'Наклонённое положение (сосредоточенность)',
                'walking': 'Идёт',
                'running': 'Бежит',
                'phone': 'Использует телефон',
                'standing': 'Стоит',
                'sitting': 'Сидит',
                'arms_crossed': 'Руки скрещены',
                'normal': 'Естественное состояние',
            },
        }

        self.emotion_map = {
            'hy': {
                'happy': 'Երջանիկ',
                'sad': 'Ծանրը',
                'neutral': 'Սթափ',
                'angry': 'Բարկացած',
                'surprised': 'Հիացած',
                'biting_lip': 'Շուրթը կծած',
            },
            'en': {
                'happy': 'Happy',
                'sad': 'Sad',
                'neutral': 'Calm',
                'angry': 'Angry',
                'surprised': 'Surprised',
                'biting_lip': 'Biting lip',
            },
            'ru': {
                'happy': 'Счастливый',
                'sad': 'Грустный',
                'neutral': 'Спокойный',
                'angry': 'Злой',
                'surprised': 'Удивлённый',
                'biting_lip': 'Кусает губу',
            },
        }

    def _action_labels(self, language):
        return self.action_map.get(language) or self.action_map['en']

    def _emotion_labels(self, language):
        return self.emotion_map.get(language) or self.emotion_map['en']

    # COCO class ids this app looks for: person (0), vehicles a
    # legal/incident video commonly involves (bicycle 1, car 2, motorcycle 3,
    # bus 5, truck 7 — e.g. a traffic stop or accident), and cell phone (67).
    # Previously only [0, 67] were requested at all, so any vehicle in frame
    # was invisible to detect_objects no matter how confidently YOLO would
    # have found it.
    _DETECTED_CLASSES = [0, 1, 2, 3, 5, 7, 67]

    def detect_objects(self, frame):
        if self.yolo is None:
            return [], []
        # conf=0.5 (ultralytics' own default is 0.25) drops the weakest,
        # spurious duplicate boxes outright — measured on real footage
        # (a person filmed close-up/off-center), plain classes=[0, 67] with
        # no confidence floor let boxes as low as 0.29 confidence through
        # alongside the real ~0.7-0.8 confidence detections of the same
        # person, one contributing factor to get_person_boxes' de-dup below
        # having so much to clean up in the first place.
        results = self.yolo(frame, verbose=False, classes=self._DETECTED_CLASSES, conf=0.5)
        detected_objects = [
            self.yolo.names[int(c)] for r in results for c in r.boxes.cls
        ]
        return results, detected_objects

    def _box_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def _box_containment(self, a, b):
        """Fraction of box a's own area that overlaps box b. Plain IoU misses
        the case where a is much smaller than b (e.g. a tight torso-only box
        sitting almost entirely inside a separate, much taller full-body box
        YOLO drew for the SAME person) — IoU divides by the large union area
        and comes out low even though a is basically a subset of b."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        return (inter / area_a) if area_a else 0.0

    def get_person_boxes(self, results):
        """YOLO's own IoU-based NMS doesn't catch every duplicate: measured on
        real footage, the SAME real person can get boxed twice at very
        different scales (a tight torso-only box AND a separate, much taller
        full-body box) — those two boxes have low IoU purely from the size
        mismatch, even though the smaller one sits almost entirely inside the
        larger one. Left uncaught, both boxes get pose-classified separately,
        double-counting one person's pose as if two different people were
        doing two different (often contradictory — e.g. one crop reads as
        "sitting", the other as "running") things. This re-ranks all person
        boxes by confidence and drops any lower-confidence box that
        substantially overlaps or is contained within an already-kept one."""
        candidates = []
        for r in results:
            for box in r.boxes:
                if self.yolo.names[int(box.cls[0])] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    candidates.append((float(box.conf[0]), (x1, y1, x2, y2)))
        candidates.sort(key=lambda c: c[0], reverse=True)

        kept = []
        for _, box in candidates:
            if not any(
                self._box_iou(box, k) > 0.3 or self._box_containment(box, k) > 0.6
                for k in kept
            ):
                kept.append(box)
        return kept

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _infer_emotion_from_landmarks(self, landmarks, image_shape, emotion_map):
        h, w = image_shape[:2]

        def pt(index):
            return np.array([landmarks[index].x * w, landmarks[index].y * h])

        left_mouth = pt(61)
        right_mouth = pt(291)
        top_lip = pt(13)
        bottom_lip = pt(14)

        mouth_width = self._distance(left_mouth, right_mouth) + 1e-6
        mouth_height = self._distance(top_lip, bottom_lip)
        smile_ratio = mouth_width / mouth_height if mouth_height else 0.0

        # Lips pressed almost flat (mouth_height collapses toward zero while
        # the corners stay at a normal talking width) reads as lip-biting/
        # pursed-lips tension, not a smile. Measured against real footage
        # (src/data/v24025gl0000d99b9hfog65qakvae190.MP4): a near-shut mouth
        # (height ~2-3px on a 960px-wide frame) pushed smile_ratio as high as
        # ~40, well past the happy check below, even though 'happy' requires
        # a genuinely open curve (mouth_height > 0.02*h) that a pressed-flat
        # mouth never reaches. Checked first since it's the more specific
        # shape — a real smile has both width AND some real height.
        if smile_ratio > 8.0 and mouth_height < 0.02 * h:
            return emotion_map['biting_lip']

        if smile_ratio > 4.0 and mouth_height > 0.02 * h:
            return emotion_map['happy']
        if mouth_height > 0.05 * h and smile_ratio < 2.5:
            return emotion_map['surprised']
        if smile_ratio < 2.0:
            return emotion_map['sad']

        # A brow-furrow 'angry' check used to live here (inner eyebrows
        # pulled toward inter-eye distance, ratio < 0.9). Removed — measured
        # directly against real footage (same video as above, 2 different
        # people, 10 sampled frames with a usable face) and it fired on
        # 10/10 of them: resting/neutral inner-brow spacing for real faces
        # sat at 0.6-0.8, i.e. already well under the "pronounced furrow"
        # threshold before any actual anger. Ordinary anatomical variation in
        # eyebrow spacing swamps a fixed per-frame threshold like this one —
        # there's no genuinely-angry sample in reach to recalibrate against
        # either, so (same call as the removed slap/push/sitting heuristics
        # above) shipping an unvalidated guess is worse than not detecting
        # 'angry' via this signal at all.
        return emotion_map['neutral']

    def detect_emotion(self, frame, language='hy'):
        """Returns an emotion label only when a face was actually found —
        None otherwise. Previously this returned emotion_map['neutral'] in
        both the "face found, no expression signal" case AND the "no face at
        all" case, so a frame with nobody in it still reported an emotion
        (e.g. "Սթափ") as if someone were there. Callers should treat None as
        "no person to read an emotion from", not skip the distinction.

        `language` selects which of self.emotion_map's per-language label
        tables ("hy"/"en"/"ru") the returned string is drawn from, falling
        back to "en" for anything else."""
        emotion_map = self._emotion_labels(language)
        if self.mp_face is not None:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(image_rgb)
            if results.multi_face_landmarks:
                return self._infer_emotion_from_landmarks(
                    results.multi_face_landmarks[0].landmark,
                    frame.shape,
                    emotion_map,
                )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            return emotion_map['neutral']
        return None

    def _is_hands_on_hips(self, lm):
        r_wrist = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        r_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        l_hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        return (
            abs(r_wrist.y - r_hip.y) < 0.1
            and abs(l_wrist.y - l_hip.y) < 0.1
            and abs(r_wrist.x - r_hip.x) > 0.1
            and abs(l_wrist.x - l_hip.x) > 0.1
        )

    def _is_pointing(self, lm):
        r_wrist = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        r_index = lm[mp.solutions.pose.PoseLandmark.RIGHT_INDEX]
        r_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        l_wrist = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        l_index = lm[mp.solutions.pose.PoseLandmark.LEFT_INDEX]
        l_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        return (
            abs(r_wrist.y - r_shoulder.y) < 0.15 and r_index.x > r_shoulder.x + 0.2
        ) or (
            abs(l_wrist.y - l_shoulder.y) < 0.15 and l_index.x < l_shoulder.x - 0.2
        )

    def _is_bent_over(self, lm):
        nose = lm[mp.solutions.pose.PoseLandmark.NOSE]
        r_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        l_hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        return nose.y > max(r_hip.y, l_hip.y) - 0.05

    def _landmarks_visible(self, lm, *names):
        """Whether every one of the given named pose landmarks individually
        meets _MIN_LANDMARK_VISIBILITY.

        _pose_is_reliable checks that 50% of *all 15* tracked landmarks are
        visible, which a close-up/upper-body-only shot (a singer at a studio
        mic, someone filmed from the chest up) can clear easily on
        wrists/shoulders/nose alone while hips/knees/ankles are out of frame
        — MediaPipe still emits *some* (x, y) for an occluded landmark, just
        not a meaningful one. That's exactly what was producing a confident
        "Running" on a video with no legs in frame at all: the aggregate
        passed on upper-body landmarks while the leg landmarks _is_running
        actually reads were noise. Each lower-body-reading heuristic below
        (hands-on-hips, bent-over, arms-crossed, walking/running/sitting —
        all of which read hip and/or knee/ankle coordinates) is gated on
        visibility of exactly the landmarks it uses, not just the aggregate,
        so an out-of-frame body part can't drive a confident-looking but
        meaningless label. Upper-body-only heuristics (pointing, phone,
        hand-up) don't need this extra gate — the landmarks they read are
        already part of the aggregate check and tend to be in frame
        together."""
        return all(
            lm[getattr(mp.solutions.pose.PoseLandmark, name)].visibility >= self._MIN_LANDMARK_VISIBILITY
            for name in names
        )

    def _is_walking(self, lm):
        r_ankle = lm[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
        l_ankle = lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
        r_knee = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
        l_knee = lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
        hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        ankle_distance = abs(r_ankle.x - l_ankle.x)
        knee_bend = ((r_knee.y + l_knee.y) / 2) > hip.y + 0.05
        return ankle_distance > 0.30 and knee_bend

    def _is_running(self, lm):
        ankle_distance = abs(
            lm[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x
            - lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x
        )
        nose = lm[mp.solutions.pose.PoseLandmark.NOSE]
        left_hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        torso_angle = abs(nose.y - ((left_hip.y + right_hip.y) / 2))
        return ankle_distance > 0.35 and torso_angle > 0.13

    def _is_sitting(self, lm):
        r_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        l_hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        r_knee = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
        l_knee = lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
        r_ankle = lm[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
        l_ankle = lm[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
        hip_y = (r_hip.y + l_hip.y) / 2
        knee_y = (r_knee.y + l_knee.y) / 2
        ankle_y = (r_ankle.y + l_ankle.y) / 2
        # Sitting: hips and knees at roughly the same height (thighs
        # horizontal) while the ankles are still clearly below the knees
        # (shins vertical) — distinguishes it from a bent-over/crouching
        # pose, where the ankles stay close to the knees too.
        return abs(hip_y - knee_y) < 0.1 and (ankle_y - knee_y) > 0.1

    def _is_arms_crossed(self, lm):
        r_wrist = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        r_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        l_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        r_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        l_hip = lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        chest_top = min(r_shoulder.y, l_shoulder.y)
        chest_bottom = max(r_hip.y, l_hip.y)
        # Each wrist has crossed over to the OPPOSITE side of the body
        # (right wrist past the left shoulder's x, and vice versa) while
        # sitting at chest height — distinct from hands-on-hips (wrists stay
        # on their own side, near the hips) and hand-up (wrists above head).
        wrists_crossed = r_wrist.x < l_shoulder.x and l_wrist.x > r_shoulder.x
        at_chest_height = (
            chest_top - 0.05 < r_wrist.y < chest_bottom + 0.05
            and chest_top - 0.05 < l_wrist.y < chest_bottom + 0.05
        )
        return wrists_crossed and at_chest_height

    def _is_phone(self, lm, detected_objects):
        if 'cell phone' in detected_objects:
            return True
        r_wrist = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        r_elbow = lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        l_elbow = lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        nose = lm[mp.solutions.pose.PoseLandmark.NOSE]
        return (
            abs(r_wrist.y - nose.y) < 0.12 and abs(r_elbow.y - nose.y) < 0.15
        ) or (
            abs(l_wrist.y - nose.y) < 0.12 and abs(l_elbow.y - nose.y) < 0.15
        )

    # Landmarks classify_actions' heuristics actually read. If MediaPipe isn't
    # confident about most of these (occlusion, motion blur, a bad crop from a
    # jittery YOLO box), the raw (x, y, z) positions are unreliable — geometric
    # checks on noise is exactly what was producing "random" action spikes.
    _POSE_LANDMARKS_USED = [
        'RIGHT_WRIST', 'LEFT_WRIST', 'NOSE', 'RIGHT_HIP', 'LEFT_HIP',
        'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE',
        'RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_INDEX', 'LEFT_INDEX',
        'RIGHT_ELBOW', 'LEFT_ELBOW',
    ]
    _MIN_LANDMARK_VISIBILITY = 0.5

    def _pose_is_reliable(self, lm):
        indices = [getattr(mp.solutions.pose.PoseLandmark, name) for name in self._POSE_LANDMARKS_USED]
        visible = sum(1 for i in indices if lm[i].visibility >= self._MIN_LANDMARK_VISIBILITY)
        # Measured directly against real footage (src/data/1.mp4, downscaled
        # to 960px width as process_video/analyze_video_headless actually do
        # it): genuine, usable pose estimates on this video clustered at
        # 0.53-0.73 visible-landmark ratio, essentially never reaching 0.7 —
        # that original threshold was rejecting almost every real frame, not
        # just the noisy ones, which is why action detection had collapsed
        # to just the generic "normal state" fallback instead of reporting
        # anything real. 0.5 still rejects clearly bad poses without
        # discarding the typical case.
        return visible / len(indices) >= 0.5

    def classify_actions(self, crop, detected_objects, language='hy'):
        action_map = self._action_labels(language)
        if self.mp_pose is None:
            return [action_map['normal']]

        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res_mp = self.mp_pose.process(image_rgb)
        if not res_mp.pose_landmarks:
            return [action_map['normal']]

        lm = res_mp.pose_landmarks.landmark
        if not self._pose_is_reliable(lm):
            return [action_map['normal']]
        actions = []
        r_wrist = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        nose = lm[mp.solutions.pose.PoseLandmark.NOSE]

        # A "slap" heuristic used to live here: right wrist above nose level,
        # roughly x-aligned with it. Removed — that's just "hand near own
        # face" (scratching, adjusting glasses, touching hair all match it
        # too), a single static frame with no motion and no check for another
        # person being nearby. It was firing on video with no slap in it and
        # labeling it with a specific criminal-code citation (Article 195),
        # which this kind of geometry can't actually support.
        #
        # A "push" heuristic used to live here too (both wrists' z well behind
        # the body plane). Removed for the same reason: measured against real
        # footage of someone just standing and gesturing while talking to the
        # camera — no push in sight — it fired on 8 of 10 sampled frames.
        # z-depth is MediaPipe's least reliable axis to begin with, and
        # "physical force" is exactly the kind of legally-loaded claim this
        # kind of unreliable geometry can't actually support.
        if r_wrist.y < (nose.y - 0.2) or l_wrist.y < (nose.y - 0.2):
            actions.append(action_map['hand_up'])
        if self._landmarks_visible(lm, 'RIGHT_HIP', 'LEFT_HIP') and self._is_hands_on_hips(lm):
            actions.append(action_map['hands_on_hips'])
        if self._is_pointing(lm):
            actions.append(action_map['pointing'])
        if self._landmarks_visible(lm, 'RIGHT_HIP', 'LEFT_HIP') and self._is_bent_over(lm):
            actions.append(action_map['bent_over'])
        if self._landmarks_visible(lm, 'RIGHT_HIP', 'LEFT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE'):
            if self._is_running(lm):
                actions.append(action_map['running'])
            elif self._is_walking(lm):
                actions.append(action_map['walking'])
            elif self._is_sitting(lm):
                actions.append(action_map['sitting'])
        if self._landmarks_visible(lm, 'RIGHT_HIP', 'LEFT_HIP') and self._is_arms_crossed(lm):
            actions.append(action_map['arms_crossed'])
        if self._is_phone(lm, detected_objects):
            actions.append(action_map['phone'])

        # A "sitting" heuristic used to live here: hip.y close to knee.y.
        # Removed — measured directly against real footage and it fired
        # almost exclusively on small/distant bounding boxes with
        # anatomically implausible landmark positions (one case had the
        # estimated ankle ABOVE the hip), plus at least one large "reliable"
        # box that still misfired. Same class of problem as the slap
        # heuristic: a single crude geometric threshold making a specific,
        # wrong claim ("Դատական նիստի կարգ" / court-session seating) with no
        # real corroborating signal.

        return actions or [action_map['standing']]
