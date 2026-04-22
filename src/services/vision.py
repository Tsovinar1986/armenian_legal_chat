import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class LegalVisionService:
    def __init__(self, state):
        self.state = state
        
        self.yolo = YOLO('yolov8n.pt') 
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Armenian Action Map
        self.action_map = {
            "slap": "Ապտակ (Ֆիզիկական բռնություն - ՀՀ քր. օր. 195 հոդված)",
            "push": "Հրում (Ֆիզիկական ներգործություն)",
            "hand_up": "Ձեռքի բարձրացում (Խոսքի իրավունքի խնդրանք)",
            "writing": "Գրառում կատարել (Փաստաթղթավորում)",
            "phone": "Հեռախոսի օգտագործում (Հնարավոր ապացույցի ձայնագրում)",
            "sitting": "Նստած (Դատական նիստի կարգ)",
            "standing": "Կանգնած (Հարգանքի դրսևորում)",
            "normal": "Բնական վիճակ"
        }
        
        # Path to a font that supports Armenian on macOS
        self.font_path = "/Library/Fonts/Arial Unicode.ttf" 

    def _draw_unicode_text(self, frame, text, position):
        """Helper to draw Armenian text using Pillow."""
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype(self.font_path, 20)
        except:
            font = ImageFont.load_default()
            
        draw.text(position, text, font=font, fill=(0, 255, 0)) # Green text
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def analyze_skeleton(self, lm, detected_objects):
        r_wrist = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        nose = lm[mp.solutions.pose.PoseLandmark.NOSE]
        r_hip = lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        r_knee = lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]

        if r_wrist.y < nose.y and abs(r_wrist.x - nose.x) < 0.15:
            return self.action_map["slap"]
        if r_wrist.z < -0.6 and l_wrist.z < -0.6:
            return self.action_map["push"]
        if r_wrist.y < (nose.y - 0.2):
            return self.action_map["hand_up"]
        if abs(r_hip.y - r_knee.y) < 0.15:
            return self.action_map["sitting"]

        return self.action_map["standing"]

    def process_frame(self, frame):
        results = self.yolo(frame, verbose=False, classes=[0, 67])
        actions_in_frame = []
        objects_seen = [self.yolo.names[int(c)] for r in results for c in r.boxes.cls]

        for r in results:
            for box in r.boxes:
                if self.yolo.names[int(box.cls[0])] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    res_mp = self.mp_pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    
                    if res_mp.pose_landmarks:
                        action = self.analyze_skeleton(res_mp.pose_landmarks.landmark, objects_seen)
                        if "cell phone" in objects_seen:
                            actions_in_frame.append(self.action_map["phone"])
                        actions_in_frame.append(action)
                        
                        # DRAWING THE ARMENIAN TEXT ON THE FRAME
                        frame = self._draw_unicode_text(frame, action, (x1, y1 - 30))

        self.state.people_actions = list(set(actions_in_frame)) 
        return frame # Return the modified frame to show in cv2.imshow