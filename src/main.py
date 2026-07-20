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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    """
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        except ImportError:
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
        print("\n📂 Enter full path to legal document (txt, xlsx) or video (mp4, mov):")
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
        # Headless (analyze_video_headless), not process_video — [u]pload is a
        # terminal action and shouldn't pop open a GUI video window; that's
        # only appropriate for the live webcam feed. Same detection pipeline
        # either way, just no cv2.imshow.
        if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            print(f"🎥 Analyzing video: {file_path}...")
            try:
                result = self.vision.analyze_video_headless(file_path)
                actions = result["actions"]
                if actions:
                    print(f"\n✅ Analysis complete ({result['frames_analyzed']} frames sampled). "
                          f"Detected unique legal actions: {actions}")
                else:
                    print(f"\n✅ Analysis complete ({result['frames_analyzed']} frames sampled). "
                          f"No legal actions were detected in the uploaded video.")
                if result["emotion"]:
                    print(f"   Emotion: {result['emotion']}")
            except Exception as ex:
                print(f"⚠️ Video analysis failed: {ex}")
            return

        # --- DOCUMENT PROCESSING ---
        print("Processing file and embedding into database...")
        status = self.ingestor.process_file(file_path)
        print(f"✅ {status}")

        try:
            if file_path.endswith('.xlsx'):
                import pandas as pd
                df = pd.read_excel(file_path)
                doc_text = df.iloc[:, 0].astype(str).str.cat(sep=' ')
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    doc_text = f.read()

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

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    try:
        client = open_persistent_client("./chroma_legal_data")
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