# 🏛️ Armenian Legal AI — Complete Project Documentation

**Version:** 1.0  
**Date:** June 2026  
**Language:** Eastern Armenian  
**Project:** Interactive AI-powered legal assistance for Armenian-language documents

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [What It Does](#what-it-does)
3. [Architecture](#architecture)
4. [Installation & Setup](#installation--setup)
5. [Core Components](#core-components)
6. [Working Code Examples](#working-code-examples)
7. [Features & Workflow](#features--workflow)
8. [Database & Storage](#database--storage)
9. [Running the Application](#running-the-application)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

**Armenian Legal AI** is a research prototype that combines multiple technologies to provide legal assistance in Eastern Armenian:

- **Retrieval-Augmented Generation (RAG):** Grounds AI responses in real court case data
- **Computer Vision:** Detects body language cues and legal actions from webcam
- **Voice Interface:** Accepts questions in Armenian via microphone and responds in Armenian
- **Knowledge Base:** Searchable database of Armenian legal cases using Chroma vector store
- **Document Processing:** Ingests `.txt`, `.xlsx`, and video files for analysis

**Target Audience:** Legal researchers, students, and those learning Armenian legal terminology  
**Disclaimer:** Not a substitute for licensed attorneys

---

## 🔧 What It Does

### 1. **Interactive Chat Interface** (`src/main.py`)
- Desktop loop with real-time webcam preview
- On-screen Armenian instructions for legal actions
- Keyboard shortcuts for easy navigation:
  - **M** = Speak a question via microphone
  - **T** = Type a question
  - **U** = Upload a legal document or video
  - **Q** = Quit

### 2. **Legal Case Analysis**
- Analyzes uploaded `.txt`, `.xlsx`, and video files
- Extracts legal concepts and terminology
- Provides Armenian legal explanations grounded in case law
- Searches vector database for similar cases

### 3. **Vision-Based Action Detection**
- Detects body poses using MediaPipe
- Recognizes objects using YOLOv8
- Maps physical actions to Armenian legal terminology
- Example: "Hand raised" → "Խոսքի իրավունքի խնդրանք" (Request to speak)

### 4. **Voice Input/Output**
- Speech Recognition: Converts microphone input to Armenian text
- Text-to-Speech: Generates Armenian audio responses
- Supports natural language queries in Armenian

### 5. **Data Ingestion Pipeline**
- Processes raw case lists from CSV/Excel
- Converts to embeddings using `nomic-embed-text`
- Stores in Chroma vector database for retrieval
- Enables fast similarity search

---

## 🏗️ Architecture

```
Armenian_Chat_part/
├── src/
│   ├── main.py                    # Entry point: Main controller loop
│   ├── analysis.py                # Statistical analysis of legal data
│   ├── Extraction_text.py         # Parse case lists to structured data
│   ├── core/
│   │   └── state.py              # Application state management
│   ├── services/
│   │   ├── vision.py             # Video processing & pose detection
│   │   ├── voice.py              # Speech recognition & text-to-speech
│   │   ├── ingestion.py          # Document & file processing
│   │   ├── classifier.py         # Legal case classification
│   │   └── case_export.py        # Export legal cases to CSV
│   ├── agents/
│   │   └── legal_agent.py        # LLM prompts & RAG logic
│   └── db/
│       └── repository.py          # Chroma vector store access
├── notebook/
│   ├── Modeling.ipynb            # ML experiments (XGBoost)
│   ├── Modeling1.ipynb           # Additional modeling
│   ├── Labeling.ipynb            # Data labeling interface
│   └── eda_armenian_full_document_text.ipynb
├── src/data/
│   ├── court_papers_full.csv     # Court case data (136MB)
│   ├── Cleaned_Verdict_Text.csv  # Cleaned verdicts (125MB)
│   ├── 1.mp4                     # Video for action detection
│   └── caseList*.txt             # Raw case lists
├── requirements.txt              # Python dependencies
├── README.md                     # Quick start guide
└── COMPREHENSIVE_DOCUMENTATION.md # This file
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10+
- Ollama with models: `nomic-embed-text` and `armenia-lawyer-router`
- Webcam (optional but recommended)
- Microphone (for voice features)

### Step 1: Clone Repository
```bash
git clone https://github.com/Tsovinar1986/armenian_legal_chat.git
cd armenian_legal_chat
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate      # Windows
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up Ollama Models
```bash
# Ensure Ollama is running
ollama serve

# In another terminal:
ollama pull nomic-embed-text
ollama pull armenia-lawyer-router
```

### Step 5: Run the Application
```bash
python src/main.py
```

---

## 🔌 Core Components

### 1. **Legal Agent** (`src/agents/legal_agent.py`)

**Purpose:** Interfaces with the Ollama LLM and retrieves relevant cases

```python
from langchain_ollama import OllamaLLM
from src.db.repository import CompanyLegalRepo

class LegalAgent:
    def __init__(self, repo, state, classifier=None, model=None):
        self.repo = repo
        self.state = state
        self.model_name = model or "armenia-lawyer-router"
        self.court_cases = []
        
        # Initialize LLM
        try:
            self.llm = OllamaLLM(model=self.model_name)
            print(f"✅ LLM initialized: {self.model_name}")
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            self.llm = None
```

**Key Methods:**
- `get_advice(question)` - Get legal advice with RAG context
- `_find_relevant_cases(query, limit=3)` - Find similar cases
- `_truncate_text(text, max_chars=900)` - Prepare context for LLM

### 2. **Vision Service** (`src/services/vision.py`)

**Purpose:** Detect legal actions from video/webcam

```python
import cv2
from ultralytics import YOLO
import mediapipe as mp

class LegalVisionService:
    def __init__(self, state):
        self.state = state
        self.yolo = YOLO('yolov8n.pt')
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Map actions to Armenian legal terminology
        self.action_map = {
            "slap": "Ապտակ (ՀՀ քր. օր. 195)",
            "push": "Հրում (Ֆիզիկական ներգործություն)",
            "hand_up": "Խոսքի իրավունքի խնդրանք",
            "sitting": "Դատական նիստի կարգ",
            "standing": "Հարգանքի դրսևորում"
        }
```

**Key Methods:**
- `process_video(video_path)` - Analyze video file for actions
- `process_frame(frame)` - Detect poses in single frame
- `_detect_actions(landmarks)` - Identify legal actions from pose

### 3. **Voice Service** (`src/services/voice.py`)

**Purpose:** Handle speech recognition and text-to-speech

```python
import speech_recognition as sr
from gtts import gTTS
import pygame

class VoiceService:
    def __init__(self, state):
        self.state = state
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        pygame.mixer.init()
    
    def listen_once(self):
        """Listen for Armenian speech and return text"""
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=10)
                text = self.recognizer.recognize_google(
                    audio, 
                    language="hy-AM"  # Eastern Armenian
                )
                return text
        except Exception as e:
            print(f"🎙️ Error: {e}")
            return None
    
    def speak(self, text: str):
        """Convert Armenian text to speech"""
        try:
            tts = gTTS(text=text, lang='hy', slow=False)
            tts.save("response.mp3")
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()
        except Exception as e:
            print(f"🔊 TTS Error: {e}")
```

### 4. **Ingestion Service** (`src/services/ingestion.py`)

**Purpose:** Process and embed documents

```python
class IngestionService:
    def process_file(self, file_path: str) -> str:
        """Process CSV, Excel, or TXT file"""
        if file_path.endswith('.xlsx'):
            # Process Excel
            import pandas as pd
            df = pd.read_excel(file_path)
            text = df.iloc[:, 0].astype(str).str.cat(sep=' ')
        else:
            # Process TXT
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Embed and store in Chroma
        self.repo.add_documents(texts=[text])
        return f"✅ Processed {len(text)} characters"
```

### 5. **Repository/Database** (`src/db/repository.py`)

**Purpose:** Manage Chroma vector store

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class CompanyLegalRepo:
    def __init__(self, collection_name="legal_cases"):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        self.vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_legal_data"
        )
    
    def search_cases(self, query: str, k: int = 3) -> list:
        """Retrieve similar cases"""
        results = self.vector_db.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    
    def add_documents(self, texts: list):
        """Add documents to vector store"""
        self.vector_db.add_texts(texts)
```

---

## 💡 Working Code Examples

### Example 1: Ask a Legal Question (Main Workflow)

```python
# From src/main.py - Handle typed question
def handle_typed_text(self):
    print("\n⌨️ Type your legal question:")
    lines = []
    while True:
        line = input()
        if line == "":  # Two enters to submit
            break
        lines.append(line)
    
    question = "\n".join(lines)
    if question.strip():
        question = unicodedata.normalize('NFC', question).strip()
        print(f"👤 You: {question}")
        
        # Get legal advice with RAG
        response = self.agent.get_advice(question)
        print(f"\n⚖️ Legal AI:\n{response}")
```

### Example 2: Process a Video (Detect Actions)

```python
# From src/main.py - Handle file upload
def handle_upload(self):
    file_path = input("📂 Enter path to document or video: ").strip()
    
    # VIDEO PROCESSING
    if file_path.lower().endswith(('.mp4', '.mov', '.avi')):
        print(f"🎥 Analyzing video: {file_path}")
        try:
            detected_actions = self.vision.process_video(file_path)
            print(f"✅ Detected actions: {detected_actions}")
        except Exception as ex:
            print(f"⚠️ Video analysis failed: {ex}")
        return
    
    # DOCUMENT PROCESSING
    status = self.ingestor.process_file(file_path)
    print(f"✅ {status}")
    
    # Get AI analysis
    response = self.agent.get_advice(open(file_path).read())
    print(f"⚖️ Legal AI Analysis:\n{response}")
```

### Example 3: Speech Recognition in Armenian

```python
# From src/services/voice.py
def listen_once(self):
    with self.microphone as source:
        self.recognizer.adjust_for_ambient_noise(source)
        audio = self.recognizer.listen(source, timeout=15)
    
    try:
        # Recognize Eastern Armenian (hy-AM)
        text = self.recognizer.recognize_google(
            audio, 
            language="hy-AM"
        )
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand Armenian speech")
        return None
```

### Example 4: Retrieve Similar Cases from Database

```python
# From src/agents/legal_agent.py
def get_advice(self, question: str) -> str:
    # Search vector database for similar cases
    similar_cases = self.repo.search_cases(question, k=3)
    
    # Build context for LLM
    context = "\n".join([
        self._truncate_text(case) for case in similar_cases
    ])
    
    # Create prompt in Armenian
    prompt = f"""
    Ներկայացված հարցը։ {question}
    
    Նմանատիպ դատական դեպքեր։
    {context}
    
    Տրամադրել մանրամասն իրավական խորհուրդ հայերեն։
    """
    
    # Get response from LLM
    response = self.llm.invoke(prompt)
    return response
```

---

## 🎮 Features & Workflow

### Interactive Workflow

```
START APPLICATION
    ↓
┌─────────────────────────────────────┐
│ Main Loop - Keyboard Input          │
├─────────────────────────────────────┤
│ M = Microphone (Voice)              │
│ T = Type Question                   │
│ U = Upload Document/Video           │
│ Q = Quit                            │
└─────────────────────────────────────┘
    │
    ├─→ [M] Voice Input
    │   ├─→ Speech Recognition (hy-AM)
    │   ├─→ Get advice from LLM + RAG
    │   └─→ Text-to-Speech Response
    │
    ├─→ [T] Typed Question
    │   ├─→ Normalize Unicode
    │   ├─→ Search vector DB
    │   ├─→ Get advice from LLM
    │   └─→ Display response
    │
    ├─→ [U] Upload File
    │   ├─→ Video (.mp4/.mov)
    │   │   └─→ Detect legal actions
    │   │       └─→ Return Armenian action names
    │   │
    │   └─→ Document (.txt/.xlsx)
    │       ├─→ Ingest to vector DB
    │       └─→ Get AI analysis
    │
    └─→ [Q] Exit
```

### Legal Case Classification

```python
from src.services.classifier import LegalCaseClassifier

classifier = LegalCaseClassifier()

# Classify a case
category = classifier.classify(case_text)
# Returns: "Վերաբերմունք & Տույժեր", "Ընտանեկան", etc.

# Get matching cases
matches = classifier.find_matching_cases(query)
# Returns: [list of similar cases]
```

---

## 📊 Database & Storage

### Vector Store (Chroma)

**Location:** `./chroma_legal_data/`  
**Capacity:** ~500k+ case documents  
**Embedding Model:** `nomic-embed-text`

```python
# Query the database
results = repo.vector_db.similarity_search(
    query="հանցագործություն",  # "crime"
    k=5  # Return top 5 matches
)

for doc in results:
    print(doc.page_content)
    print(f"Score: {doc.metadata.get('similarity')}")
```

### Data Files (`src/data/`)

| File | Size | Purpose |
|------|------|---------|
| `court_papers_full.csv` | 136 MB | All court cases |
| `Cleaned_Verdict_Text.csv` | 125 MB | Cleaned verdict text |
| `legal_analysis_labeled.csv` | 125 MB | Labeled for ML |
| `verdicts.txt` | 126 MB | Raw verdict text |
| `1.mp4` | 71 MB | Sample video for analysis |

---

## 🚀 Running the Application

### Mode 1: Interactive Desktop App

```bash
python src/main.py
```

**Features:**
- Live webcam preview
- Real-time pose detection
- Voice input/output
- Document upload
- Keyboard shortcuts (M, T, U, Q)

### Mode 2: Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `notebook/` folder:
- `Modeling.ipynb` - ML experiments
- `eda_armenian_full_document_text.ipynb` - Data exploration
- `Labeling.ipynb` - Manual data labeling

### Mode 3: Command-Line Analysis

```python
# Extract and analyze cases
python src/Extraction_text.py

# Generate statistics
python src/analysis.py
```

---

## 🛠️ Troubleshooting

### Issue: "Ollama model not found"

```bash
# Ensure Ollama is running
ollama serve

# Pull missing model
ollama pull armenia-lawyer-router
ollama pull nomic-embed-text
```

### Issue: "Microphone not detected"

```python
# List available microphones when prompted
# Choose correct device number
# Or manually set in code:
mic = sr.Microphone(device_index=1)  # Try different index
```

### Issue: "Cannot import pynput"

```bash
# Optional dependency for keyboard shortcuts
pip install pynput

# App still works without it (keyboard hotkeys disabled)
```

### Issue: "PyAudio installation failed"

```bash
# macOS
brew install portaudio
pip install PyAudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install PyAudio
```

### Issue: "Vision features not working"

```bash
# Reinstall MediaPipe
pip install --upgrade mediapipe==0.10.30

# Or ensure correct version
pip install "mediapipe>=0.10.21,<0.10.31"
```

---

## 📚 Key Technologies

| Component | Technology | Version |
|-----------|-----------|---------|
| **LLM** | Ollama + OllamaLLM | Latest |
| **Embeddings** | nomic-embed-text | Latest |
| **Vector DB** | Chroma | 0.4+ |
| **Vision** | YOLOv8, MediaPipe | Latest |
| **Speech** | Google Speech API, gTTS | 3.10+, 2.90+ |
| **ML** | scikit-learn, XGBoost | 1.4+, 2.0+ |
| **Data** | Pandas, NumPy | 2.0+, 1.26+ |

---

## 📝 Development Notes

### Adding New Features

1. **New Service:** Create in `src/services/`
2. **New Agent Logic:** Extend `src/agents/legal_agent.py`
3. **New UI Commands:** Add keyboard handler in `src/main.py`

### Customizing Armenian Content

All Armenian text is configurable:

```python
# src/services/vision.py - Action names
self.action_map = {
    "custom_action": "Հայերեն բացատրություն (Օրենք հղ.)"
}

# src/agents/legal_agent.py - Prompts
prompt = "Հայերեն հարց..."
```

### Performance Optimization

```python
# Limit database searches
similar_cases = repo.search_cases(query, k=3)  # Not 20

# Cache results
@functools.lru_cache(maxsize=100)
def get_cached_advice(question):
    return agent.get_advice(question)
```

---

## 📞 Support & Contact

**Repository:** https://github.com/Tsovinar1986/armenian_legal_chat  
**Issues:** Report bugs on GitHub Issues  
**Documentation:** See `README.md` for quick start

---

## ⚖️ Disclaimer

This project is for **research and educational purposes** only. It does not constitute legal advice. Always consult a licensed attorney for real legal matters. The outputs are generated by machine learning models and may be incomplete or incorrect.

---

**Last Updated:** June 2026  
**Maintained by:** Armenian Legal AI Project Team
