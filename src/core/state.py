from threading import Lock

class SystemState:
    def __init__(self):
        self._lock = Lock()
        self.is_running = True
        self.webcam_active = True
        self.mic_active = True
        self.current_action = None
        
        # Guard flag to pause background threads during user interactive text input
        self.terminal_input_active = False

        # Data shared between Vision and Agent
        self.people_actions = []  # Armenian action names
        self.current_emotion = "Neutral"
        # Non-person object class names (YOLO COCO labels) seen in the most
        # recent frame — lets callers distinguish "no person, but an object
        # was detected" from "nothing detected at all" instead of collapsing
        # both into a fake/default action or emotion. See LegalVisionService.
        self.detected_objects = []
        self.active_category = "General"
        self.file_context = ""

        # Set when LegalVisionService observes a sustained negative facial-affect
        # pattern (see src/services/vision.py). This is a soft UX nudge, not a
        # diagnosis — it only suggests talking to a therapist, it never claims to
        # detect a clinical condition.
        self.mental_health_concern = False
        self.mental_health_suggestion = ""

    def update_actions(self, actions):
        with self._lock:
            self.people_actions = actions

    def update_emotion(self, emotion):
        with self._lock:
            self.current_emotion = emotion

    def update_objects(self, objects):
        with self._lock:
            self.detected_objects = objects

    def update_category(self, category):
        with self._lock:
            self.active_category = category

    def update_context(self, context):
        with self._lock:
            self.file_context = context

    def update_mental_health_concern(self, flag: bool, suggestion: str = ""):
        with self._lock:
            self.mental_health_concern = flag
            self.mental_health_suggestion = suggestion if flag else ""

    def get_actions(self):
        with self._lock:
            return self.people_actions.copy()

    def get_emotion(self):
        with self._lock:
            return self.current_emotion

    def get_objects(self):
        with self._lock:
            return self.detected_objects.copy()

    def get_category(self):
        with self._lock:
            return self.active_category

    def get_context(self):
        with self._lock:
            return self.file_context

    def get_mental_health_concern(self):
        with self._lock:
            return self.mental_health_concern

    def get_mental_health_suggestion(self):
        with self._lock:
            return self.mental_health_suggestion
