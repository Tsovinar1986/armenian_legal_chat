import sys
import os
import cv2
from pynput import keyboard
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.core.state import SystemState
    from src.services.vision import LegalVisionService
    from src.services.voice import VoiceService
    from src.services.ingestion import IngestionService
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
        if os.path.exists(file_path):
            print("Processing file...")
            status = self.ingestor.process_file(file_path)
            print(f"✅ {status}")
        else:
            print("⚠️ File not found. Please check the path.")

    def handle_mic(self):
        """Trigger manual voice input"""
        user_speech = self.voice.listen_once()
        if user_speech:
            print(f"\n👤 You: {user_speech}")
            response = self.agent.get_advice(user_speech)
            print(f"⚖️ Legal AI: {response}")
            self.voice.speak(response)


def main():
    print("⚖️ Armenian Legal AI System Starting...\n")
    
    state = SystemState()
    state.webcam_active = True
    state.is_running = True

    # Initialize RAG components
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient(path="./chroma_legal_data")
    vector_db = Chroma(
        collection_name="company_legal_cases",
        embedding_function=embeddings,
        client=client
    )

    # Initialize services
    voice_service = VoiceService(state)                    # ← This will show mic list
    vision_service = LegalVisionService(state)
    legal_agent = LegalAgent(CompanyLegalRepo(vector_db), state)
    ingestor = IngestionService(vector_db)

    controller = LegalAIController(state, vision_service, voice_service, legal_agent, ingestor)

    # Start background voice listener
    voice_service.start_background_listener()

    # Keyboard controls
    def on_press(key):
        try:
            if hasattr(key, 'char'):
                if key.char == 'm':
                    controller.handle_mic()
                elif key.char == 'u':
                    controller.handle_upload()
                elif key.char == 'v':
                    print("👁️ Vision mode toggle (coming soon)")
                elif key.char == 'q':
                    state.is_running = False
                    return False
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("\n🎮 CONTROLS:")
    print("   [m] → Speak (Microphone)")
    print("   [u] → Upload legal document")
    print("   [v] → Vision mode (future)")
    print("   [q] → Quit\n")

    # Main Vision + Window Loop
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
        print("\n👋 Goodbye! System stopped.")

if __name__ == "__main__":
    main()
