# project-edited
# 🤖 TutoAI – AI-Powered Learning from Your Documents

TutoAI is an AI-powered document learning assistant that allows users to upload multiple document formats, build a Retrieval-Augmented Generation (RAG) knowledge base, ask questions, generate quizzes, review weak topics using spaced repetition, and visualize source citations.

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

### 📝 Quiz Generator

Generate:

- MCQs
- Descriptive Questions

Choose:

- Topic
- Number of Questions

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

---

## 📁 Project Structure

```
TutoAI/
│
├── app.py
├── rag_engine1.py
├── progress_store.py
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

---

## ▶️ Run

```bash
streamlit run app.py
```

---

## ⚙️ Environment Variables

Create a `.env` file.

### Ollama

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
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
7. Review weak topics
8. Track mastery

---

## 📸 Screenshots

Add screenshots of:

- Home Page
- Chat Interface
- Dashboard
- Quiz Generator
- Review Due
- Mastery Dashboard

---

## 🚀 Future Enhancements

- Voice Assistant
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
