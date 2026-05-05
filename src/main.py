import sys
import os
import cv2
from pynput import keyboard 

# --- Path Configuration ---
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
        """Manual trigger for file ingestion."""
        print("\n📂 [TRIGGER] Enter full path to legal file:")
        file_path = input(">>> ").strip()
        if os.path.exists(file_path):
            status = self.ingestor.process_file(file_path)
            print(f"✅ Status: {status}")
        else:
            print("⚠️ File not found.")

    def handle_mic(self):
        """Manual trigger for voice processing."""
        print("\n🎤 [TRIGGER] Listening now...")
        user_speech = self.voice.listen_once() 
        if user_speech:
            print(f"👤 User: {user_speech}")
            response = self.agent.get_advice(user_speech)
            print(f"⚖️ AI: {response}")
            self.voice.speak(response)

def main():
    print("⚖️ Armenian Legal AI (Webcam Active)")
    
    # 1. Resource-Efficient Initialization
    state = SystemState()
    # CHANGED: Set webcam to True by default
    state.webcam_active = True 
    state.is_running = True

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient(path="./chroma_legal_data")
    vector_db = Chroma(collection_name="company_legal_cases", embedding_function=embeddings, client=client)
    
    controller = LegalAIController(
        state, 
        LegalVisionService(state), 
        VoiceService(state), 
        LegalAgent(CompanyLegalRepo(vector_db), state),
        IngestionService(vector_db)
    )

    # 2. Keyboard Listener Thread
    def on_press(key):
        try:
            if key.char == 'u': controller.handle_upload()
            if key.char == 'm': controller.handle_mic()
            if key.char == "v": controller.toggle_vision()
            if key.char == 'q': 
                state.is_running = False
                return False 
        except AttributeError: pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("\n⌨️  CONTROLS:")
    print(" [m] Use Mic | [u] Upload File | [v] Toggle Vision | [q] Quit")

    # 3. Vision Loop
    cap = None 
    window_name = "Legal AI Feed"

    try:
        while state.is_running:
            # The loop now assumes webcam is always intended to be on
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            ret, frame = cap.read()
            if ret:
                frame = controller.vision.process_frame(frame)
                cv2.imshow(window_name, frame)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                state.is_running = False

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System Shutdown Cleanly.")

if __name__ == "__main__":
    main()