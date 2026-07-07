import sys
import os
import time
import unicodedata
import cv2

# Ensure the project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    from langchain_chroma import Chroma
    import chromadb
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
        file_path = input(">>> ").strip().strip('"\'')
        if not os.path.exists(file_path):
            print("⚠️ File not found.")
            return

        # --- VIDEO PROCESSING PIPELINE ---
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

    def handle_similar_cases(self):
        query = input("\n🔍 Describe your case for search: ").strip()
        if query:
            cases = self.agent.get_similar_cases(query, limit=5)
            if cases: print(f"\n{self.agent.format_similar_cases_response(cases)}")

    def handle_approved_cases(self):
        result = self.agent.get_approved_cases_with_lawyers(limit=20)
        if result.get('approved_cases'): print(f"\n{self.agent.format_approved_cases_response(result)}")


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
    client = chromadb.PersistentClient(path="./chroma_legal_data")
    vector_db = Chroma(collection_name="company_legal_cases", embedding_function=embeddings, client=client)
    
    voice_service = VoiceService(state)
    vision_service = LegalVisionService(state)
    classifier_service = LegalCaseClassifier(data_folder="src/data")
    legal_agent = LegalAgent(CompanyLegalRepo(vector_db), state, classifier=classifier_service)
    ingestor = IngestionService(vector_db)

    controller = LegalAIController(state, vision_service, voice_service, legal_agent, ingestor)
    voice_service.start_background_listener()

    def on_press(key):
        if not state.terminal_input_active:
            try:
                if hasattr(key, 'char') and key.char in ['m', 't', 'u', 's', 'a', 'q']:
                    state.current_action = key.char
            except: pass

    listener = None
    if keyboard is not None:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("\n🎮 CONTROLS: [m]ic, [t]ype, [u]pload (doc/video), [s]imilar, [a]pproved, [q]uit")
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
                elif action == 's': controller.handle_similar_cases()
                elif action == 'a': controller.handle_approved_cases()
                elif action == 'q': state.is_running = False
                state.terminal_input_active = False

            if state.webcam_active:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(state.camera_source)
                    if not cap.isOpened():
                        print(f"⚠️ Cannot open camera source: {state.camera_source}")
                        time.sleep(1)
                        continue

                ret, frame = cap.read()
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