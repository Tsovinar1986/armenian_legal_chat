# src/main_qwen.py — A/B comparison variant of src/main.py, identical except
# for one line: LegalAgent is constructed with model="qwen3" instead of the
# default "armenia-lawyer-router". Everything else (vision, voice, RAG
# pipeline, crisis detection, guardrails) is unchanged — this exists purely to
# let you compare answer quality/style between the two Ollama models on the
# same case data and prompts.
#
# Before running, pull the model once:
#   ollama pull qwen3
# (or a specific size tag, e.g. `ollama pull qwen3:8b` — larger tags are
# slower but generally higher quality; edit QWEN_MODEL_NAME below to match
# whichever tag you pulled.)
#
# Run with: python src/main_qwen.py
#
# See README.md "Qwen vs. armenia-lawyer-router" for how to interpret the
# comparison and whether Qwen has any usage cost.
QWEN_MODEL_NAME = "qwen3"

import sys
import os
import time
import unicodedata
import cv2

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
    import chromadb
    from src.db.vector_store import ChromaVectorStore
except ImportError as e:
    print(f"❌ Critical import failed: {e}")
    print("Please install required packages with: pip install -r requirements.txt")
    sys.exit(1)

from src.main import LegalAIController


def main():
    print(f"⚖️ Armenian Legal AI System Starting (model: {QWEN_MODEL_NAME})...\n")

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

    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # unchanged — the vector DB was indexed with this
    client = chromadb.PersistentClient(path="./chroma_legal_data")
    vector_db = ChromaVectorStore(client=client, collection_name="company_legal_cases", embeddings=embeddings)

    voice_service = VoiceService(state)
    vision_service = LegalVisionService(state)
    classifier_service = LegalCaseClassifier(data_folder="src/data")
    legal_agent = LegalAgent(CompanyLegalRepo(vector_db), state, classifier=classifier_service, model=QWEN_MODEL_NAME)
    ingestor = IngestionService(vector_db)

    controller = LegalAIController(state, vision_service, voice_service, legal_agent, ingestor)
    voice_service.start_background_listener()

    def on_press(key):
        if not state.terminal_input_active:
            try:
                if hasattr(key, 'char') and key.char in ['m', 't', 'u', 'q']:
                    state.current_action = key.char
            except: pass

    listener = None
    if keyboard is not None:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        print("\n🎮 CONTROLS: [m]ic, [t]ype, [u]pload (doc/video), [q]uit")
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
