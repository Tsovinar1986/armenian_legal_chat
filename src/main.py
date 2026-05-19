import sys
import os
import threading

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
    def __init__(self, state, vision, voice, agent, ingestor, classifier):
        self.state = state
        self.vision = vision
        self.voice = voice
        self.agent = agent
        self.ingestor = ingestor
        self.classifier = classifier

    def handle_upload(self):
        def upload_worker():
            print("\n📂 Enter full path to legal document:")
            file_path = input(">>> ").strip().strip('"\'')
            if not os.path.exists(file_path):
                print("⚠️ File not found. Please check the path.")
                return

            print("Processing file...")
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

                print("🔍 Analyzing case patterns and looking for historical precedents...")
                matched_case = self.classifier.find_similar_case(doc_text)

                if matched_case:
                    print("\n🎯 [CLASSIFIER MATCH FOUND]")
                    print(f"   🔹 Classifier: {matched_case.get('civil_case_classifier')}")
                    print(f"   🔹 Similar case: {matched_case.get('unique_number')}")
                    print(f"   🔹 Link: {matched_case.get('link')}")
                    lawyer = matched_case.get('lawyer_name')
                    lawyer_display = lawyer if lawyer and lawyer != "(NULL)" else "Not specified"
                    print(f"   🔹 Suggested lawyer: {lawyer_display}")
                else:
                    print("\nℹ️ No similar classified historical precedent found.")
            except Exception as ex:
                print(f"⚠️ Error during case classification: {ex}")

        threading.Thread(target=upload_worker, daemon=True).start()

    def handle_mic(self):
        def mic_worker():
            user_speech = self.voice.listen_once()
            if user_speech:
                print(f"\n👤 You: {user_speech}")
                response = self.agent.get_advice(user_speech)
                print(f"⚖️ Legal AI:\n{response}")
                self.voice.speak(response)

        threading.Thread(target=mic_worker, daemon=True).start()


def main():
    print("⚖️ Armenian Legal AI System Starting...\n")

    state = SystemState()
    state.webcam_active = True
    state.is_running = True

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
    legal_agent = LegalAgent(CompanyLegalRepo(vector_db), state)
    ingestor = IngestionService(vector_db)

    controller = LegalAIController(
        state, vision_service, voice_service, legal_agent, ingestor, classifier_service
    )

    try:
        voice_service.start_background_listener()
    except Exception as e:
        print(f"⚠️ Background listener failed: {e}")

    def on_press(key):
        try:
            if hasattr(key, 'char') and key.char:
                if key.char == 'm':
                    controller.handle_mic()
                elif key.char == 'u':
                    controller.handle_upload()
                elif key.char == 'q':
                    state.is_running = False
                    return False
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press, suppress=False)
    listener.start()

    print("\n🎮 CONTROLS:")
    print("   [m] → Speak (Microphone)")
    print("   [u] → Upload legal document (with classifier matching)")
    print("   [q] → Quit\n")

    cap = None
    window_name = "Legal AI Feed"

    try:
        while state.is_running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            ret, frame = cap.read()
            if ret:
                processed_frame = vision_service.process_frame(frame)
                cv2.imshow(window_name, processed_frame)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                state.is_running = False
                break
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        listener.stop()
        print("\n👋 Goodbye! System stopped.")


if __name__ == "__main__":
    main()
