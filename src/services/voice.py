import speech_recognition as sr
import pyttsx3
import queue

class VoiceService:
    def __init__(self, state):
        # FIXED: Accept and store the shared state
        self.state = state
        self.input_queue = queue.Queue()
        
        # Initialize Text-to-Speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def speak(self, text):
        """Converts AI response text to Armenian speech."""
        print(f"🔊 AI Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_loop(self):
        """Background thread that listens for Armenian speech."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("🎤 Microphone ready... (Listening for Armenian)")
            
            while self.state.is_running:
                try:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=5)
                    # Use 'hy-AM' for Armenian language recognition
                    text = self.recognizer.recognize_google(audio, language="hy-AM")
                    if text:
                        self.input_queue.put(text)
                except sr.UnknownValueError:
                    pass # Ignore noise
                except Exception as e:
                    print(f"🎙️ Voice Error: {e}")