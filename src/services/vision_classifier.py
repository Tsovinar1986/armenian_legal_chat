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

        self.action_map = {
            'slap': 'Ապտակ (Ֆիզիկական բռնություն - ՀՀ քր. օր. 195 հոդված)',
            'push': 'Հրում (Ֆիզիկական ներգործություն)',
            'hand_up': 'Ձեռքի բարձրացում (Խոսքի իրավունքի խնդրանք)',
            'hands_on_hips': 'Ձեռքեր գոտկատեղին (Պաշտպանողական դիրք)',
            'pointing': 'Ցույց տալ (Ուղղորդող նշում)',
            'bent_over': 'Ծռված դիրք (Ուշադրության կենտրոնացում)',
            'walking': 'Քայլում է (Շարժման ացքը)',
            'running': 'Վազում է (Արագ շարժում)',
            'phone': 'Հեռախոսի օգտագործում (Հնարավոր ապացույցի ձայնագրում)',
            'sitting': 'Նստած (Դատական նիստի կարգ)',
            'standing': 'Կանգնած (Հարգանքի դրսևորում)',
            'normal': 'Բնական վիճակ',
        }

        self.emotion_map = {
            'happy': 'Երջանիկ',
            'sad': 'Ծանրը',
            'neutral': 'Սթափ',
            'angry': 'Բարկացած',
            'surprised': 'Հիացած',
        }

    def detect_objects(self, frame):
        if self.yolo is None:
            return [], []
        results = self.yolo(frame, verbose=False, classes=[0, 67])
        detected_objects = [
            self.yolo.names[int(c)] for r in results for c in r.boxes.cls
        ]
        return results, detected_objects

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _infer_emotion_from_landmarks(self, landmarks, image_shape):
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

        if smile_ratio > 4.0 and mouth_height > 0.02 * h:
            return self.emotion_map['happy']
        if mouth_height > 0.05 * h and smile_ratio < 2.5:
            return self.emotion_map['surprised']
        if smile_ratio < 2.0:
            return self.emotion_map['sad']
        return self.emotion_map['neutral']

    def detect_emotion(self, frame):
        """Returns an emotion label only when a face was actually found —
        None otherwise. Previously this returned emotion_map['neutral'] in
        both the "face found, no expression signal" case AND the "no face at
        all" case, so a frame with nobody in it still reported an emotion
        (e.g. "Սթափ") as if someone were there. Callers should treat None as
        "no person to read an emotion from", not skip the distinction."""
        if self.mp_face is not None:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(image_rgb)
            if results.multi_face_landmarks:
                return self._infer_emotion_from_landmarks(
                    results.multi_face_landmarks[0].landmark,
                    frame.shape,
                )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            return self.emotion_map['neutral']
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
        return visible / len(indices) >= 0.7

    def classify_actions(self, crop, detected_objects):
        if self.mp_pose is None:
            return [self.action_map['normal']]

        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res_mp = self.mp_pose.process(image_rgb)
        if not res_mp.pose_landmarks:
            return [self.action_map['normal']]

        lm = res_mp.pose_landmarks.landmark
        if not self._pose_is_reliable(lm):
            return [self.action_map['normal']]
        actions = []
        r_wrist = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        nose = lm[mp.solutions.pose.PoseLandmark.NOSE]
        r_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        r_knee = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]

        if r_wrist.y < nose.y and abs(r_wrist.x - nose.x) < 0.15:
            actions.append(self.action_map['slap'])
        if r_wrist.z < -0.6 and l_wrist.z < -0.6:
            actions.append(self.action_map['push'])
        if r_wrist.y < (nose.y - 0.2) or l_wrist.y < (nose.y - 0.2):
            actions.append(self.action_map['hand_up'])
        if self._is_hands_on_hips(lm):
            actions.append(self.action_map['hands_on_hips'])
        if self._is_pointing(lm):
            actions.append(self.action_map['pointing'])
        if self._is_bent_over(lm):
            actions.append(self.action_map['bent_over'])
        if self._is_running(lm):
            actions.append(self.action_map['running'])
        elif self._is_walking(lm):
            actions.append(self.action_map['walking'])
        if self._is_phone(lm, detected_objects):
            actions.append(self.action_map['phone'])
        if abs(r_hip.y - r_knee.y) < 0.15:
            actions.append(self.action_map['sitting'])

        return actions or [self.action_map['standing']]
