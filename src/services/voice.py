import re
import speech_recognition as sr
from gtts import gTTS
import pygame
import queue
import threading
import time
import os


# recognize_google occasionally mishears an English loanword said mid-Armenian
# sentence as a similar-sounding but unrelated phrase (e.g. "MasterCard" -> "master
# cartoon"). Known cases get corrected here; keyed by the mistaken phrase
# (case-insensitive), mapped to the intended term.
_KNOWN_MISHEARINGS = {
    "master cartoon": "MasterCard",
    "master car tune": "MasterCard",
    "mastercartoon": "MasterCard",
    "visa cart": "Visa card",
}


def _fix_known_mishearings(text: str) -> str:
    for wrong, right in _KNOWN_MISHEARINGS.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text


def sanitize_transcript(text: str, language: str = "hy") -> str:
    """Drop empty/near-empty or junk that recognize_google occasionally
    returns for silence or noise. Shared by VoiceService (mic loop) and
    api.py's /api/speech-to-text (browser-recorded audio) so both entry
    points filter the same way.

    The token-repetition checks are language-agnostic (\\w already matches
    Armenian/Cyrillic/Latin letters). The "require an Armenian letter unless
    the transcript is long" check only makes sense when hy-AM is actually
    the language being recognized — applying it to en/ru transcripts would
    silently drop every short, valid "Hello"/"Привет"-style utterance.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = _fix_known_mishearings(cleaned)

    lower = cleaned.lower()
    tokens = re.findall(r"[\wЀ-ӿ]+", lower)

    if len(tokens) >= 3 and len(set(tokens)) == 1 and len(tokens[0]) <= 3:
        return ""

    if len(tokens) >= 3 and all(len(t) <= 3 for t in tokens):
        if len(set(tokens)) == 2 and len(tokens) == 3:
            return ""

    if language == "hy" and not re.search(r'[ա-ֆԱ-Ֆ]', cleaned):
        if len(cleaned) < 20:
            return ""

    return cleaned


class VoiceService:
    def __init__(self, state):
        self.state = state
        self.input_queue = queue.Queue()

        # Initialize Pygame Mixer for natural audio streaming playback
        pygame.mixer.init()

        # Speech Recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        print("✅ Using default microphone")

        self.listening_thread = None

    def speak(self, text: str):
        """Generates clear, beautiful native Armenian voice answers instead of robotic sound."""
        print(f"🔊 AI: {text}")
        if not text.strip():
            return

        temp_filename = "temp_response.mp3"
        try:
            # Generate high-quality audio payload targeting Armenian ('hy') dialect
            tts = gTTS(text=text, lang='hy', slow=False)
            tts.save(temp_filename)

            # Play back audio file seamlessly
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()

            # Prevent thread from crashing while playback finishes
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.unload()
        except Exception as e:
            print(f"⚠️ Audio streaming playback engine issue: {e}")
        finally:
            # Housekeeping file cleaning safely wrapped
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass

    def _sanitize_transcript(self, text: str) -> str:
        return sanitize_transcript(text)

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
            # Completely pause listening iteration if the terminal input guard is active
            # or the user has toggled the mic off.
            if getattr(self.state, 'terminal_input_active', False) or not getattr(self.state, 'mic_active', True):
                time.sleep(0.2)
                continue

            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=6)

                # Secondary validation check before launching web translation requests
                if getattr(self.state, 'terminal_input_active', False):
                    continue

                text = self.recognizer.recognize_google(audio, language="hy-AM")
                text = self._sanitize_transcript(text)
                if text:
                    self.input_queue.put(text)
                    print(f"📥 Heard: {text}")
            except Exception:
                time.sleep(0.3)