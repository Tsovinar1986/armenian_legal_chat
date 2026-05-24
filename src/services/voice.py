import re
import speech_recognition as sr
import pyttsx3
import queue
import threading
import time

class VoiceService:
    def __init__(self, state):
        self.state = state
        self.input_queue = queue.Queue()

        # Text-to-Speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        # Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.current_device_index = None

        self.listening_thread = None
        self.list_available_microphones()

    def list_available_microphones(self):
        """List all microphones and let user choose (especially useful for headphones)."""
        print("\n🎤 Available Microphones:")
        mic_list = sr.Microphone.list_microphone_names()

        for i, name in enumerate(mic_list):
            print(f"  [{i}] {name}")

        print("\n💡 Plug in your headphones/earphones now if you want to use their mic.")

        try:
            choice = input("Enter microphone number (or press Enter for default): ").strip()
            if choice.isdigit():
                self.current_device_index = int(choice)
                self.microphone = sr.Microphone(device_index=self.current_device_index)
                print(f"✅ Selected: {mic_list[self.current_device_index]}")
            else:
                self.microphone = sr.Microphone()
                print("✅ Using default microphone")
        except Exception as e:
            print(f"⚠️ Using default mic. Error: {e}")
            self.microphone = sr.Microphone()

    def speak(self, text: str):
        print(f"🔊 AI: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def _sanitize_transcript(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""

        lower = cleaned.lower()
        tokens = re.findall(r"[\w\u0400-\u04FF]+", lower)

        # Ignore short repeated babble such as 'fg fg fe' or 'aa aa aa'
        if len(tokens) >= 3 and len(set(tokens)) == 1 and len(tokens[0]) <= 3:
            return ""

        # Ignore repeated short token patterns like 'fg fg fe'
        if len(tokens) >= 3 and all(len(t) <= 3 for t in tokens):
            if len(set(tokens)) == 2 and len(tokens) == 3:
                return ""

        # Accept Armenian text, but reject short non-Armenian noise
        if not re.search(r'[ա-ֆԱ-Ֆ]', cleaned):
            if len(cleaned) < 20:
                return ""

        return cleaned

    def listen_once(self) -> str:
        """Manual one-time listen."""
        if not self.microphone:
            self.microphone = sr.Microphone()

        print("🎤 Listening... Speak now")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                audio = self.recognizer.listen(source, timeout=7, phrase_time_limit=10)

            text = self.recognizer.recognize_google(audio, language="hy-AM")
            text = self._sanitize_transcript(text)
            if not text:
                print("❌ Sorry, I didn't catch valid speech. Please try again.")
                return ""

            print(f"👤 You said: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ Sorry, I didn't catch that.")
            return ""
        except sr.RequestError as e:
            print(f"❌ Google Speech service error: {e}")
            return ""
        except Exception as e:
            print(f"🎙️ Mic error: {e}")
            return ""

    def start_background_listener(self):
        if self.listening_thread and self.listening_thread.is_alive():
            return
        self.listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listening_thread.start()
        print("🎤 Background voice listener started")

    def _listen_loop(self):
        print("🎤 Background listening ready...")
        while self.state.is_running:
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=6)

                text = self.recognizer.recognize_google(audio, language="hy-AM")
                text = self._sanitize_transcript(text)
                if text:
                    self.input_queue.put(text)
                    print(f"📥 Heard: {text}")
            except Exception:
                time.sleep(0.3)
