import sys
import os
import threading
import cv2
from pynput import keyboard  # High-efficiency key listener

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
        self.cap = None

    def handle_upload(self):
        """Manual trigger for ingestion."""
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
        # Uses your voice service for a single capture
        user_speech = self.voice.listen_once() 
        if user_speech:
            print(f"👤 User: {user_speech}")
            response = self.agent.get_advice(user_speech)
            print(f"⚖️ AI: {response}")
            self.voice.speak(response)

    def toggle_vision(self):
        """Toggles webcam to save RAM/CPU."""
        self.state.webcam_active = not getattr(self.state, 'webcam_active', False)
        mode = "ON" if self.state.webcam_active else "OFF"
        print(f"\n📷 [TRIGGER] Vision Mode: {mode}")
        if not self.state.webcam_active:
            cv2.destroyAllWindows()

def main():
    print("⚖️ Armenian Legal AI (Memory Optimized for Mac)")
    
    # 1. Resource-Efficient Initialization
    state = SystemState()
    state.webcam_active = False 
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

    # 2. Keyboard Listener Thread (Very low RAM usage)
    def on_press(key):
        try:
            if key.char == 'u': controller.handle_upload()
            if key.char == 'm': controller.handle_mic()
            if key.char == 'v': controller.toggle_vision()
            if key.char == 'q': 
                state.is_running = False
                return False 
        except AttributeError: pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("\n⌨️  CONTROLS:")
    print(" [V] Toggle Webcam | [M] Use Mic | [U] Upload File | [Q] Quit")

    # 3. Conditional Vision Loop
    cap = cv2.VideoCapture(0)
    
    try:
        while state.is_running:
            if getattr(state, 'webcam_active', False):
                ret, frame = cap.read()
                if ret:
                    controller.vision.process_frame(frame)
                    # Overlay only if active to save processing
                    actions = " | ".join(state.people_actions) if state.people_actions else "Passive"
                    cv2.putText(frame, f"AI Vision: {actions}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Legal AI Feed", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Idle state: reduce CPU usage to near 0%
                cv2.waitKey(200) 

    finally:
        state.is_running = False
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System Shutdown Cleanly.")

if __name__ == "__main__":
    main()