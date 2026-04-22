from threading import Lock

class SystemState:
    def __init__(self):
        self._lock = Lock()
        self.is_running = True
        
        # Data shared between Vision and Agent
        self.people_actions = []  # Armenian action names
        self.current_emotion = "Neutral"
        self.active_category = "General"
        self.file_context = ""

    def update_actions(self, actions):
        with self._lock:
            self.people_actions = actions