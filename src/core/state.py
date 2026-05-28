from threading import Lock

class SystemState:
    def __init__(self):
        self._lock = Lock()
        self.is_running = True
        self.webcam_active = True
        self.current_action = None
        
        # Guard flag to pause background threads during user interactive text input
        self.terminal_input_active = False

        # Data shared between Vision and Agent
        self.people_actions = []  # Armenian action names
        self.current_emotion = "Neutral"
        self.active_category = "General"
        self.file_context = ""

    def update_actions(self, actions):
        with self._lock:
            self.people_actions = actions

    def update_emotion(self, emotion):
        with self._lock:
            self.current_emotion = emotion

    def update_category(self, category):
        with self._lock:
            self.active_category = category

    def update_context(self, context):
        with self._lock:
            self.file_context = context

    def get_actions(self):
        with self._lock:
            return self.people_actions.copy()

    def get_emotion(self):
        with self._lock:
            return self.current_emotion

    def get_category(self):
        with self._lock:
            return self.active_category

    def get_context(self):
        with self._lock:
            return self.file_context    