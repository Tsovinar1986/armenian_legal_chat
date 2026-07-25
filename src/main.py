# src/main.py — the DESKTOP CLI app: webcam + microphone loop, keyboard-driven
# (m/t/u/q controls). Run with: python src/main.py
#
# This is a different entry point from api.py at the repo root, which is the
# FastAPI WEB PORTAL (browser chat, REST API, WebRTC). They share the same
# underlying LegalAgent/classifier/vector-store code in src/, but are two
# separate ways to run this project — not two versions of the same file.
# Run the web portal with: uvicorn api:app --reload
import sys
import os
import platform
import time
import unicodedata
import cv2

# Ensure the project root is in path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_PROJECT_ROOT)


def _flush_stdin():
    """Discard input typed but not yet read.

    The pynput global keyboard listener (see on_press below) doesn't consume
    the keystroke it reacts to — the same physical keypress (e.g. 'u' to
    trigger [u]pload) also lands in the terminal's own stdin buffer. That
    stray character then sits there until the next input() call, which
    silently reads it as its first character — e.g. a path typed right after
    pressing 'u' arrives as "u/real/path", which then fails the file-exists
    check even though the real path is correct. Call this immediately before
    any input() prompt that follows a keyboard-triggered action.

    Windows note: pynput's low-level keyboard hook doesn't suppress the
    keystroke either, so the same leak happens there. The correct Windows
    equivalent of termios.tcflush is FlushConsoleInputBuffer (a Win32 API
    call, via ctypes) — it clears the console's raw input buffer directly,
    which is where a leaked keystroke actually lands. msvcrt.kbhit()/getch()
    read from that same buffer one key at a time and were tried first here,
    but can race with the C runtime's own line-buffering layer already having
    claimed the character before the loop gets to poll for it; going straight
    to the Win32 call is more reliable, so it's the primary path now.
    """
    if platform.system() == "Windows":
        try:
            import ctypes
            STD_INPUT_HANDLE = -10
            handle = ctypes.windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
            ctypes.windll.kernel32.FlushConsoleInputBuffer(handle)
        except Exception:
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()
            except ImportError:
                pass
        return

    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


keyboard = None
try:
    from pynput import keyboard as pynput_keyboard
    keyboard = pynput_keyboard
except ImportError:
    print("⚠️ Optional dependency 'pynput' is not installed. Keyboard shortcuts will be disabled.")

try:
    from src.core.state import SystemState
    from src.services.vision import LegalVisionService
    from src.services.voice import VoiceService
    from src.services.ingestion import IngestionService
    from src.services.classifier import LegalCaseClassifier
    from src.agents.legal_agent import LegalAgent
    from src.db.repository import CompanyLegalRepo
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

try:
    from langchain_ollama import OllamaEmbeddings
    import chromadb
    from src.db.vector_store import ChromaVectorStore, open_persistent_client
except ImportError as e:
    print(f"❌ Critical import failed: {e}")
    print("Please install required packages with: pip install -r requirements.txt")
    sys.exit(1)

class LegalAIController:
    def __init__(self, state, vision, voice, agent, ingestor):
        self.state = state
        self.vision = vision
        self.voice = voice
        self.agent = agent
        self.ingestor = ingestor
        self.conversation_history = []

    def handle_upload(self):
        print("\n📂 Enter full path to legal document (txt, xlsx, csv, json, docx, pdf, or other text-based format) or video (mp4, mov):")
        _flush_stdin()
        raw_input_path = input(">>> ")
        # Strip straight AND curly/smart quotes (macOS text substitution turns
        # ' into ‘/’ and " into “/” in some input sources) — .strip('"\'')
        # alone leaves a smart quote attached, which silently breaks
        # os.path.exists() on an otherwise-correct, genuinely existing path.
        file_path = raw_input_path.strip().strip("\"'‘’“”")
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path!r} (received: {raw_input_path!r})")
            return

        # --- VIDEO PROCESSING PIPELINE ---
        # process_video opens a playback window with a live overlay (downscaled
        # + throttled — see its docstring — to stay usable on an 8GB-RAM
        # machine) rather than analyzing headless with no visual feedback.
        if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            print(f"🎥 Analyzing video: {file_path}. Press 'q' to stop.")
            try:
                detected_actions = self.vision.process_video(file_path)
                if detected_actions:
                    print(f"\n✅ Analysis complete. Detected unique legal actions: {detected_actions}")
                else:
                    print("\n✅ Analysis complete. No legal actions were detected in the uploaded video.")
            except Exception as ex:
                print(f"⚠️ Video analysis failed: {ex}")
            return

        # --- DOCUMENT PROCESSING ---
        print("Processing file and embedding into database...")
        status = self.ingestor.process_file(file_path)
        print(f"✅ {status}")

        try:
            doc_text = self.ingestor.extract_text(file_path)
            doc_text = unicodedata.normalize('NFC', doc_text)
            response = self.agent.get_advice(doc_text)
            print(f"\n⚖️ Legal AI Analysis:\n{response}")
        except Exception as ex:
            print(f"⚠️ Error: {ex}")

    def handle_mic(self):
        if not self.state.mic_active:
            print("\n🎙️ Microphone is off. Press [v] to turn it on first.")
            return
        print("\n🤖 AI: Ինչպե՞ս կարող եմ օգնել ձեզ այսօր...")
        try:
            user_speech = self.voice.listen_once()
            if user_speech:
                user_speech = unicodedata.normalize('NFC', user_speech).strip()
                print(f"\n👤 You: {user_speech}")
                response = self.agent.get_advice(user_speech, self.conversation_history)
                print(f"\n⚖️ Legal AI:\n{response}")
                self.conversation_history.append({"role": "user", "text": user_speech})
                self.conversation_history.append({"role": "bot", "text": response})
        except Exception as e:
            print(f"🎙️ Mic error: {e}")

    def handle_typed_text(self):
        print("\n⌨️ Type your legal question. Press ENTER twice to submit:")
        _flush_stdin()
        lines = []
        while True:
            line = input(">>> " if not lines else "... ")
            if line.strip() == "":
                if lines: break
                else: return
            lines.append(line)
        user_input = unicodedata.normalize('NFC', " ".join(lines).strip())
        if user_input:
            response = self.agent.get_advice(user_input, self.conversation_history)
            print(f"\n⚖️ Legal AI:\n{response}")
            self.conversation_history.append({"role": "user", "text": user_input})
            self.conversation_history.append({"role": "bot", "text": response})



def _open_camera(source):
    """cv2.VideoCapture wrapper that picks a working backend per OS and never
    raises: on Windows, the default auto-detected backend (often MSMF) can
    hang or throw for webcam indices where DirectShow (CAP_DSHOW) works fine,
    so prefer CAP_DSHOW there for integer sources. Network/IP camera sources
    (URL strings) go through the default backend on every OS."""
    try:
        if platform.system() == "Windows" and isinstance(source, int):
            return cv2.VideoCapture(source, cv2.CAP_DSHOW)
        return cv2.VideoCapture(source)
    except Exception as exc:
        print(f"⚠️ Could not open camera source {source}: {exc}")
        return None


def main():
    print("⚖️ Armenian Legal AI System Starting...\n")

    state = SystemState()
    state.is_running = True
    
    use_webcam = input("🎥 Enable webcam or network camera stream? (y/n): ").strip().lower() == 'y'
    state.webcam_active = use_webcam
    state.camera_source = 0
    if state.webcam_active:
        source_input = input("📡 Camera source (0 for laptop webcam, or URL for IP/mobile stream): ").strip()
        if source_input:
            try:
                state.camera_source = int(source_input)
            except ValueError:
                state.camera_source = source_input

    from src.services.ollama_setup import ensure_ollama_models
    ensure_ollama_models()

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    try:
        client = open_persistent_client(os.path.join(_PROJECT_ROOT, "chroma_legal_data"))
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    vector_db = ChromaVectorStore(client=client, collection_name="company_legal_cases", embeddings=embeddings)
    
    voice_service = VoiceService(state)
    vision_service = LegalVisionService(state)
    classifier_service = LegalCaseClassifier(data_folder="src/data")
    legal_agent = LegalAgent(CompanyLegalRepo(vector_db), state, classifier=classifier_service)
    ingestor = IngestionService(vector_db)

    controller = LegalAIController(state, vision_service, voice_service, legal_agent, ingestor)

    use_mic = input("🎤 Enable background microphone listening? (y/n): ").strip().lower() == 'y'
    state.mic_active = use_mic
    if state.mic_active:
        voice_service.start_background_listener()

    def on_press(key):
        if not state.terminal_input_active:
            try:
                if hasattr(key, 'char') and key.char in ['m', 't', 'u', 'q', 'v']:
                    state.current_action = key.char
            except: pass

    listener = None
    if keyboard is not None:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("\n🎮 CONTROLS: [m]ic, [t]ype, [u]pload (doc/video), [v]oice on/off, [q]uit")
    else:
        print("\n✅ Keyboard listener disabled. Use the main app interface for input.")

    cap = None
    try:
        while state.is_running:
            if state.current_action:
                action = state.current_action
                state.current_action = None
                state.terminal_input_active = True
                if action == 'm': controller.handle_mic()
                elif action == 't': controller.handle_typed_text()
                elif action == 'u': controller.handle_upload()
                elif action == 'v':
                    state.mic_active = not state.mic_active
                    if state.mic_active:
                        voice_service.start_background_listener()
                        print("\n🎤 Microphone turned ON")
                    else:
                        print("\n🔇 Microphone turned OFF")
                elif action == 'q': state.is_running = False
                state.terminal_input_active = False

            if state.webcam_active:
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = _open_camera(state.camera_source)
                    if cap is None or not cap.isOpened():
                        print(f"⚠️ Cannot open camera source: {state.camera_source}")
                        time.sleep(1)
                        continue

                try:
                    ret, frame = cap.read()
                except Exception as exc:
                    print(f"⚠️ Camera read failed, will retry: {exc}")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue

                if ret:
                    cv2.imshow("Legal AI Feed", vision_service.process_frame(frame))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    state.is_running = False
            else:
                time.sleep(0.1)
    finally:
        if cap: cap.release()
        cv2.destroyAllWindows()
        if listener is not None:
            listener.stop()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()