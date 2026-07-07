import cv2
import numpy as np
import unicodedata
from PIL import Image, ImageDraw, ImageFont


class LegalVisionService:
    def __init__(self, state):
        self.state = state
        self._classifier = None
        self.font_path = "/Library/Fonts/Arial Unicode.ttf"

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
            cv2.putText(
                processed_frame,
                status_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, processed_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return list(unique_actions)

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

        self.state.update_actions(list(set(actions_in_frame)))
        return frame


if __name__ == "__main__":
    print("LegalVisionService module loaded.")
