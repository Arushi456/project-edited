# project-edited
# 🤖 TutoAI – AI-Powered Learning from Your Documents

TutoAI is an AI-powered document learning assistant that allows users to upload multiple document formats, build a Retrieval-Augmented Generation (RAG) knowledge base, ask questions, generate quizzes, take AI-graded voice vivas, review weak topics using spaced repetition, and track performance analytics — all backed by source citations.

Built using **Python**, **Streamlit**, **ChromaDB**, **Sentence Transformers**, and **Ollama/Gemini**.

---

## ✨ Features

### 📂 Multi-Format Document Support
- PDF
- DOCX
- PPTX
- TXT
- Markdown
- RTF
- Images (JPG, PNG, JPEG, BMP, TIFF)

---

### 🔍 OCR Support
Automatically extracts text from:
- Scanned PDFs
- Images

using **Tesseract OCR**.

---

### 🧠 AI Question Answering
Ask questions about uploaded documents using Retrieval-Augmented Generation (RAG).

Supports:
- Brief Answers
- Detailed Answers
- Beginner-Friendly Explanations
- Summaries
- Key Points

---

### 📖 Visual Citations
Every answer contains source citations.
For PDF documents, TutoAI highlights the exact section used to generate the answer.

---

### 📊 Dashboard
View analytics including:
- Uploaded documents
- Indexed chunks
- Total pages
- Detected topics
- Topic frequency

---

### 📄 Document Tools
- Generate document summaries
- Compare two documents
- Export summaries

---

### 🛣️ AI Learning Path Generator
Generate a personalized study plan based on:
- Topic
- Current Level (Beginner / Intermediate / Advanced)
- Learning Goal
- Duration (in weeks)

---

### 📝 Quiz Generator
Generate:
- MCQs
- Descriptive Questions

Choose:
- Topic
- Number of Questions

---

### 🎤 AI Viva (Voice-Enabled Oral Exam)
Take an AI-graded oral viva straight from your uploaded documents.

Setup options:
- Difficulty (Beginner / Intermediate / Advanced)
- Number of Questions
- **Topic** (optional — leave blank to draw questions from the whole document set, or set it to ground questions in a specific topic, same retrieval used by the Quiz Generator)
- Student Name (for the report)

Voice controls:
- **Speak Question** — questions are narrated aloud using `edge-tts` neural voices
- **Record Answer** — record your spoken answer in-browser; it's transcribed with `SpeechRecognition` and graded automatically

Each answer is scored out of 10 with a confidence/accuracy/completeness/communication breakdown, and question generation automatically retries if the model returns fewer valid questions than requested.

---

### 📈 Viva Analytics
A dedicated analytics tab summarizing your entire viva history:
- Total Viva Sessions
- Questions Answered
- Average / Highest / Lowest Score
- Strongest & Weakest Topic *(derived from average score per difficulty level, since individual viva questions aren't topic-tagged)*
- Score Trend & Confidence Trend charts
- Recent Viva Table
- **AI Performance Insights** — an on-demand, AI-generated summary of your strengths, weaknesses, and a concrete improvement tip

---

### 🧭 AI Study Recommendations
Blends your Quiz Generator/Review Due performance with your AI Viva performance into a single, on-demand study plan — highlighting priority topics to review next and a suggested next action.

---

### 🔁 Spaced Repetition Learning
Implements the **SM-2 Algorithm**.

Features:
- Tracks incorrect answers
- Automatically schedules review sessions
- Review Due section

---

### 📈 Mastery Dashboard
Track learning progress with:
- Topic Accuracy
- Weak Topics
- Quiz Performance

---

### 💬 Chat History
- Saves previous conversations
- Feedback system
- Export answers as TXT and PDF

---

### 🗑 Document Management
- Upload multiple documents
- Delete saved documents
- Reset Knowledge Base
- Clear Chat History

---

## 🛠️ Technologies Used
- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Ollama
- Google Gemini
- PyPDF
- python-docx
- python-pptx
- pytesseract
- Pillow
- Pandas
- dotenv
- edge-tts *(viva question narration)*
- SpeechRecognition *(viva answer transcription)*
- pydub *(audio conversion for viva answers — requires `ffmpeg` on PATH)*

---

## 📁 Project Structure
```
TutoAI/
│
├── app1.py
├── rag_engine1.py
├── progress_store.py
├── viva.py
├── requirements.txt
├── README.md
│
├── data/
├── persist/
├── models/
│
└── assets/
```

---

## ⚙️ Installation

### Clone Repository
```bash
git clone https://github.com/your-username/TutoAI.git
cd TutoAI
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install ffmpeg (required for AI Viva voice answers)
The AI Viva feature converts recorded browser audio to WAV using `pydub`, which shells out to `ffmpeg`. Install it and make sure it's on your PATH:
- **Windows:** [ffmpeg.org](https://ffmpeg.org/download.html) (or `choco install ffmpeg`)
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

---

## ▶️ Run
```bash
streamlit run app1.py
```

---

## ⚙️ Environment Variables
Create a `.env` file.

### Ollama
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

### Gemini
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-2.0-flash
```

---

## 📌 Workflow
1. Upload documents
2. Click **Ingest & Train**
3. Build the vector database
4. Ask questions
5. View cited sources
6. Generate quizzes
7. Take a voice-enabled AI Viva
8. Review weak topics
9. Track mastery
10. Check Viva Analytics and AI Study Recommendations for a personalized progress summary

---

## 📸 Screenshots
Add screenshots of:
- Home Page
- Chat Interface
- Dashboard
- Quiz Generator
- AI Viva
- Viva Analytics
- AI Study Recommendations
- Review Due
- Mastery Dashboard

---

## 🚀 Future Enhancements
- Multi-language Support
- Cloud Database
- User Authentication
- Mobile Application
- Collaborative Learning

---
Final Year B.Tech (Computer Science)

---

## 📄 License
This project is developed for educational and academic purposes.
