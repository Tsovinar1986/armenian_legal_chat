import sys
import os
import threading
import time
import unicodedata  # For resolving Armenian Unicode matching bugs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'NONE'

import cv2
from pynput import keyboard

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    print("Please check your folder structure and file names.")
    sys.exit(1)

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb


class LegalAIController:
    def __init__(self, state, vision, voice, agent, ingestor):
        self.state = state
        self.vision = vision
        self.voice = voice
        self.agent = agent
        self.ingestor = ingestor

    def handle_upload(self):
        print("\n📂 Enter full path to legal document:")
        file_path = input(">>> ").strip().strip('"\'')
        if not os.path.exists(file_path):
            print("⚠️ File not found. Please check the path.")
            return

        print("Processing file and embedding into database...")
        status = self.ingestor.process_file(file_path)
        print(f"✅ {status}")

        try:
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    doc_text = f.read()
            elif file_path.endswith('.xlsx'):
                import pandas as pd
                df = pd.read_excel(file_path)
                doc_text = df.iloc[:, 0].astype(str).str.cat(sep=' ')
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    doc_text = f.read()

            # Normalize document text to standard Form C format
            doc_text = unicodedata.normalize('NFC', doc_text)

            print("🔍 Analyzing case patterns using Centralized Agent...")
            response = self.agent.get_advice(doc_text)
            print(f"\n⚖️ Legal AI Analysis:\n{response}")
            
        except Exception as ex:
            print(f"⚠️ Error during document pipeline routing: {ex}")

    def handle_mic(self):
        print("\n🤖 AI: Ինչպե՞ս կարող եմ օգնել ձեզ այսօր: Խնդրում եմ ներկայացրեք ձեր իրավական հարցը...")
        try:
            user_speech = self.voice.listen_once()
            if user_speech:
                user_speech = unicodedata.normalize('NFC', user_speech).strip()
                print(f"\n👤 You: {user_speech}")
                print("🔍 Querying LLM and ChromaDB Vector Storage...")
                response = self.agent.get_advice(user_speech)
                print(f"\n⚖️ Legal AI:\n{response}")
            else:
                print("⚠️ No audio detected.")
                print("💡 Suggestion: If you don't have a microphone/headphones available,")
                print("   press [t] to type your legal question instead.")
        except Exception as mic_err:
            print(f"🎙️ Mic capture encountered an error: {mic_err}")
            print("💡 Troubleshooting tips:")
            print("   • Make sure you selected an Input Microphone device, not output speakers")
            print("   • Without mic/headphones? Press [t] to type your question instead")
            print("   • Or press [u] to upload a legal document for analysis")

    def handle_typed_text(self):
        print("\n⌨️ Type or paste your legal question/case description.")
        print("👉 Press ENTER twice on a blank line to complete pasting and submit:")
        
        lines = []
        while True:
            try:
                line = input(">>> " if not lines else "... ")
                if line.strip() == "":
                    if lines:
                        break
                    else:
                        print("⚠️ Cancelled empty text input.")
                        return
                lines.append(line)
            except EOFError:
                break
                
        # Combine lines and strictly normalize Unicode characters for accurate matching
        raw_input = " ".join(lines).strip()
        user_input = unicodedata.normalize('NFC', raw_input)
        
        if user_input:
            print(f"\n⚡ Input Payload Verified ({len(user_input)} characters).")
            print("🔍 Querying LLM and ChromaDB Vector Storage...")
            try:
                # Active document tracker debug hook
                if hasattr(self.agent, 'repo') and hasattr(self.agent.repo, 'vector_db'):
                    count = self.agent.repo.vector_db._collection.count()
                    print(f"📡 [DEBUG] Active documents inside collection: {count}")
                
                print("👤 Processing your text question with AI agent...")
                response = self.agent.get_advice(user_input)
                print(f"\n⚖️ Legal AI:\n{response}")
                print("\n✨ Response generated using LLM model.")
            except Exception as e:
                print(f"💥 Error retrieving agent response: {e}")
                print("💡 Tip: Check that the model and vector database are properly configured.")
        else:
            print("⚠️ Input string was empty.")


def main():
    print("⚖️ Armenian Legal AI System Starting...\n")

    if not os.path.exists("data"):
        try:
            os.makedirs("data")
            print("📁 Created missing 'data' directory.")
        except Exception as e:
            print(f"⚠️ Could not create data directory: {e}")

    state = SystemState()
    state.is_running = True
    state.current_action = None 
    state.terminal_input_active = False 

    use_webcam_input = input("🎥 Would you like to enable the webcam feed? (y/n): ").strip().lower()
    if use_webcam_input in ['y', 'yes']:
        state.webcam_active = True
        print("✅ Webcam feed will be initialized.")
    else:
        state.webcam_active = False
        print("❌ Webcam feed disabled.")

    # 🎯 Set fine-tuned model identifier explicitly
    FINETUNED_MODEL_NAME = "armenian-lawyer-router" 

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient(path="./chroma_legal_data")
    vector_db = Chroma(
        collection_name="company_legal_cases",
        embedding_function=embeddings,
        client=client
    )

    voice_service = VoiceService(state)
    vision_service = LegalVisionService(state)
    classifier_service = LegalCaseClassifier(data_folder="data")
    
    # Try injecting our custom fine-tuned variant securely into the constructor setup
    try:
        legal_agent = LegalAgent(
            CompanyLegalRepo(vector_db), 
            state, 
            classifier=classifier_service,
            model=FINETUNED_MODEL_NAME  
        )
    except TypeError:
        # If your LegalAgent constructor doesn't take 'model' directly as an arg, set it as an attribute property
        legal_agent = LegalAgent(CompanyLegalRepo(vector_db), state, classifier=classifier_service)
        legal_agent.model_name = FINETUNED_MODEL_NAME

    ingestor = IngestionService(vector_db)

    controller = LegalAIController(
        state, vision_service, voice_service, legal_agent, ingestor
    )

    try:
        voice_service.start_background_listener()
    except Exception as e:
        print(f"⚠️ Background listener failed initialization: {e}")

    def on_press(key):
        if getattr(state, 'terminal_input_active', False):
            return True
            
        try:
            if hasattr(key, 'char') and key.char:
                if key.char in ['m', 't', 'u', 'q']:
                    state.current_action = key.char
                    if key.char == 'q':
                        state.is_running = False
                        return False
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press, suppress=False)
    listener.start()

    print("\n🎮 CONTROLS:")
    print("   [m] → Speak (Microphone)")
    print("   [t] → Type your question manually")
    print("   [u] → Upload legal document (File path)")
    print("   [q] → Quit\n")

    cap = None
    window_name = "Legal AI Feed"

    if state.webcam_active:
        print("🎥 Initializing webcam feed...")

    try:
        while state.is_running:
            if state.current_action:
                action = state.current_action
                state.current_action = None 
                
                state.terminal_input_active = True
                try:
                    if action == 'm':
                        controller.handle_mic()
                    elif action == 't':
                        controller.handle_typed_text()
                    elif action == 'u':
                        controller.handle_upload()
                finally:
                    state.terminal_input_active = False

            if state.webcam_active:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    time.sleep(0.5) 
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        try:
                            processed_frame = vision_service.process_frame(frame)
                            cv2.imshow(window_name, processed_frame)
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                        except Exception as vision_err:
                            print(f"⚠️ Vision processing frame error: {vision_err}")
                    else:
                        time.sleep(0.03) 
                else:
                    time.sleep(0.03)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    state.is_running = False
                    break
            else:
                time.sleep(0.05)
                
    except Exception as loop_error:
        print(f"💥 Critical error in main loop: {loop_error}")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        listener.stop()
        print("\n👋 Goodbye! System stopped.")


if __name__ == "__main__":
    main()