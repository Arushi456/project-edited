import importlib
import io
import json
import os
import re
import textwrap
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import rag_engine1 as rag_engine_module
print("Loaded from:", rag_engine_module.__file__)
print("compare_documents exists:", hasattr(rag_engine_module, "compare_documents"))
import progress_store
progress_store = importlib.reload(progress_store)

from viva import (
    speak,
    listen,
    load_progress,
    save_progress,
    reset_progress,
    current_question,
    next_question,
    add_feedback,
    add_confidence,
    add_score,
    viva_percentage,
)

rag_engine = importlib.reload(rag_engine_module)


DATA_DIR = rag_engine.DATA_DIR
PERSIST_DIR = rag_engine.PERSIST_DIR
build_index = rag_engine.build_index
clear_data = rag_engine.clear_data
compare_documents = rag_engine.compare_documents
generate_document_summary = rag_engine.generate_document_summary
generate_learning_path = rag_engine.generate_learning_path
generate_viva_questions = rag_engine.generate_viva_questions
evaluate_viva_answer = rag_engine.evaluate_viva_answer
extract_score = rag_engine.extract_score
analyze_viva_confidence = rag_engine.analyze_viva_confidence
generate_viva_insights = rag_engine.generate_viva_insights
generate_study_recommendations = rag_engine.generate_study_recommendations
generate_quiz = rag_engine.generate_quiz
get_assistant_modes = rag_engine.get_assistant_modes
get_citation_image = rag_engine.get_citation_image
get_db_collection = rag_engine.get_db_collection
get_document_analytics = rag_engine.get_document_analytics
has_indexed_data = rag_engine.has_indexed_data
list_uploaded_pdfs = rag_engine.list_uploaded_pdfs
list_uploaded_documents = rag_engine.list_uploaded_documents
detect_document_type = rag_engine.detect_document_type
SUPPORTED_EXTENSIONS = rag_engine.SUPPORTED_EXTENSIONS
query_knowledge_base = rag_engine.query_knowledge_base
save_uploaded_file = rag_engine.save_uploaded_file
CHAT_HISTORY_PATH = os.path.join(PERSIST_DIR, "chat_history.json")

# Extensions accepted by the uploader, without the leading dot (what
# st.file_uploader's `type=` argument expects).
UPLOAD_EXTENSIONS = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)

# Small icon per file type, used in the sidebar document list and the
# analytics table so users can tell formats apart at a glance.
FILE_TYPE_ICONS = {
    "pdf": "📕",
    "docx": "📘",
    "pptx": "📙",
    "txt": "📄",
    "md": "📝",
    "rtf": "📄",
    "jpg": "🖼️",
    "jpeg": "🖼️",
    "png": "🖼️",
    "bmp": "🖼️",
    "tif": "🖼️",
    "tiff": "🖼️",
}


def file_type_icon(filename: str) -> str:
    ext = detect_document_type(filename).lstrip(".")
    return FILE_TYPE_ICONS.get(ext, "📄")

load_dotenv(override=True)


st.set_page_config(
    page_title="TutoAI Knowledge Agent",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
            :root {
                --panel: rgba(15, 23, 42, 0.82);
                --panel-soft: rgba(15, 23, 42, 0.58);
                --line: rgba(148, 163, 184, 0.2);
                --text: #e5edf8;
                --muted: #9fb0c4;
                --teal: #2dd4bf;
                --amber: #f6c45f;
                --rose: #fb7185;
                --blue: #60a5fa;
            }

            .stApp {
                color: var(--text);
                background:
                    radial-gradient(circle at 18% 16%, rgba(45, 212, 191, 0.16), transparent 28%),
                    radial-gradient(circle at 86% 8%, rgba(96, 165, 250, 0.14), transparent 26%),
                    linear-gradient(135deg, #070b15 0%, #101827 48%, #121725 100%);
            }

            .main .block-container {
                max-width: 1320px;
                padding-top: 1.5rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span {
                color: #d8e4f2;
            }

            .tuto-sidebar-brand {
                border-bottom: 1px solid var(--line);
                margin-bottom: 1rem;
                padding: 1rem 0 0.7rem;
            }

            .brand-title {
                color: #f8fafc;
                font-size: 1.25rem;
                font-weight: 800;
            }

            .brand-caption {
                color: var(--muted);
                font-size: 0.88rem;
                margin-top: 0.15rem;
            }

            .tuto-section-title {
                align-items: center;
                color: #f8fafc;
                display: flex;
                font-size: 0.82rem;
                font-weight: 800;
                gap: 0.5rem;
                letter-spacing: 0;
                margin: 1.1rem 0 0.65rem;
                text-transform: uppercase;
            }

            .tuto-hero,
            .tuto-panel,
            .metric-card,
            .source-card-link,
            .topic-card,
            .guide-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 8px;
                box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
            }

            .tuto-hero {
                background:
                    linear-gradient(135deg, rgba(20, 184, 166, 0.2), rgba(96, 165, 250, 0.08)),
                    rgba(15, 23, 42, 0.72);
                margin-bottom: 1.15rem;
                overflow: hidden;
                padding: 1.5rem;
                position: relative;
            }

            .tuto-hero:before {
                background: linear-gradient(90deg, var(--teal), var(--blue), var(--amber));
                content: "";
                height: 3px;
                left: 0;
                position: absolute;
                right: 0;
                top: 0;
            }

            .tuto-kicker {
                color: var(--teal);
                font-size: 0.78rem;
                font-weight: 800;
                margin-bottom: 0.45rem;
                text-transform: uppercase;
            }

            .tuto-hero h1 {
                color: #f8fafc;
                font-size: clamp(2rem, 4vw, 3.4rem);
                font-weight: 850;
                letter-spacing: 0;
                line-height: 1;
                margin: 0;
            }

            .tuto-hero p {
                color: #c6d3e1;
                font-size: 1.02rem;
                line-height: 1.55;
                margin: 0.85rem 0 0;
                max-width: 800px;
            }

            .tuto-status-pill,
            .mode-badge,
            .page-badge,
            .relevance-badge {
                align-items: center;
                border-radius: 999px;
                display: inline-flex;
                font-size: 0.78rem;
                font-weight: 800;
                gap: 0.35rem;
                letter-spacing: 0;
                padding: 0.28rem 0.62rem;
            }

            .tuto-status-pill {
                background: rgba(45, 212, 191, 0.1);
                border: 1px solid rgba(45, 212, 191, 0.28);
                color: #bff8ef;
                margin-top: 1rem;
            }

            .metric-card {
                min-height: 136px;
                padding: 1rem;
            }

            .metric-top {
                align-items: center;
                display: flex;
                justify-content: space-between;
                gap: 0.7rem;
            }

            .metric-icon {
                align-items: center;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                color: #f8fafc;
                display: inline-flex;
                height: 2.2rem;
                justify-content: center;
                width: 2.2rem;
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 750;
                text-transform: uppercase;
            }

            .metric-value {
                color: #f8fafc;
                font-size: 2rem;
                font-weight: 850;
                line-height: 1.1;
                margin-top: 0.8rem;
                word-break: break-word;
            }

            .metric-caption {
                color: #aebdd0;
                font-size: 0.86rem;
                margin-top: 0.35rem;
            }

            .metric-card.teal { border-top: 3px solid var(--teal); }
            .metric-card.amber { border-top: 3px solid var(--amber); }
            .metric-card.blue { border-top: 3px solid var(--blue); }
            .metric-card.rose { border-top: 3px solid var(--rose); }

            .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
            .stTabs [data-baseweb="tab"] {
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid var(--line);
                border-radius: 8px;
                color: #dbeafe;
                height: 2.7rem;
                padding: 0 1rem;
            }
            .stTabs [aria-selected="true"] {
                background: rgba(45, 212, 191, 0.12);
                border-color: rgba(45, 212, 191, 0.45);
                color: #f8fafc;
            }

            [data-testid="stChatMessage"] {
                background: rgba(15, 23, 42, 0.65);
                border: 1px solid rgba(148, 163, 184, 0.17);
                border-radius: 8px;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
                margin-bottom: 0.8rem;
                padding: 0.75rem;
            }

            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
                background: rgba(37, 99, 235, 0.16);
                border-color: rgba(96, 165, 250, 0.24);
            }

            .source-grid {
                display: grid;
                gap: 0.7rem;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                margin-top: 0.85rem;
            }

            .source-card-link {
                color: inherit;
                display: block;
                padding: 0.85rem;
                text-decoration: none;
                transition: border-color 160ms ease, transform 160ms ease;
            }

            .source-card-link:hover {
                border-color: rgba(45, 212, 191, 0.55);
                transform: translateY(-1px);
            }

            .source-card-head {
                align-items: center;
                display: flex;
                justify-content: space-between;
                gap: 0.6rem;
            }

            .source-file {
                color: #f8fafc;
                font-size: 0.92rem;
                font-weight: 750;
                line-height: 1.3;
                overflow-wrap: anywhere;
            }

            .page-badge {
                background: rgba(96, 165, 250, 0.12);
                border: 1px solid rgba(96, 165, 250, 0.28);
                color: #bfdbfe;
                white-space: nowrap;
            }

            .source-snippet {
                color: #b9c7d8;
                font-size: 0.84rem;
                line-height: 1.45;
                margin-top: 0.65rem;
            }

            .relevance-row {
                align-items: center;
                display: flex;
                gap: 0.6rem;
                margin-top: 0.7rem;
            }

            .relevance-badge {
                background: rgba(45, 212, 191, 0.1);
                border: 1px solid rgba(45, 212, 191, 0.25);
                color: #99f6e4;
            }

            .relevance-track {
                background: rgba(148, 163, 184, 0.18);
                border-radius: 999px;
                flex: 1;
                height: 0.42rem;
                overflow: hidden;
            }

            .relevance-fill {
                background: linear-gradient(90deg, var(--teal), var(--blue));
                height: 100%;
            }

            .mode-badge {
                background: rgba(246, 196, 95, 0.1);
                border: 1px solid rgba(246, 196, 95, 0.26);
                color: #fde68a;
                margin: 0 0 0.6rem;
                text-transform: uppercase;
            }

            [data-testid="stSegmentedControl"] {
                background: rgba(15, 23, 42, 0.52);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 8px;
                margin-bottom: 1rem;
                padding: 0.7rem;
            }

            [data-testid="stPills"] {
                background: rgba(15, 23, 42, 0.52);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 8px;
                margin-bottom: 0.7rem;
                padding: 0.7rem;
            }

            .tuto-panel,
            .topic-card,
            .guide-card {
                padding: 1rem;
            }

            .topic-card {
                background: var(--panel-soft);
                min-height: 92px;
            }

            .topic-title {
                color: #f8fafc;
                font-weight: 800;
                text-transform: capitalize;
            }

            .topic-meta,
            .guide-card p {
                color: var(--muted);
                font-size: 0.88rem;
                margin: 0.35rem 0 0;
            }

            .context-block {
                background: rgba(2, 6, 23, 0.42);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 8px;
                margin: 0.65rem 0;
                padding: 0.85rem;
            }

            .context-title {
                color: #f8fafc;
                font-weight: 800;
                margin-bottom: 0.35rem;
            }

            .context-text {
                color: #cbd5e1;
                font-size: 0.88rem;
                line-height: 1.5;
                overflow-wrap: anywhere;
                white-space: normal;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 8px;
                font-weight: 750;
            }

            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #14b8a6, #2563eb);
                border: 0;
            }

            .stChatInput textarea,
            [data-testid="stAlert"] {
                border-radius: 8px;
            }

            @media (max-width: 720px) {
                .main .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                .metric-card { min-height: auto; }
                .tuto-hero { padding: 1.1rem; }
            }
            .viva-badge-row {
                align-items: center;
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin: 0.6rem 0 0;
            }

            .difficulty-badge {
                border-radius: 999px;
                border: 1px solid;
                font-size: 0.76rem;
                font-weight: 800;
                padding: 0.26rem 0.65rem;
                text-transform: uppercase;
            }

            .difficulty-badge.beginner {
                background: rgba(45, 212, 191, 0.12);
                border-color: rgba(45, 212, 191, 0.32);
                color: #99f6e4;
            }

            .difficulty-badge.intermediate {
                background: rgba(246, 196, 95, 0.12);
                border-color: rgba(246, 196, 95, 0.32);
                color: #fde68a;
            }

            .difficulty-badge.advanced {
                background: rgba(251, 113, 133, 0.12);
                border-color: rgba(251, 113, 133, 0.32);
                color: #fecdd3;
            }

            .viva-counter-badge {
                background: rgba(96, 165, 250, 0.12);
                border: 1px solid rgba(96, 165, 250, 0.3);
                border-radius: 999px;
                color: #bfdbfe;
                font-size: 0.76rem;
                font-weight: 800;
                padding: 0.26rem 0.65rem;
            }

            .viva-progress-track {
                background: rgba(148, 163, 184, 0.18);
                border-radius: 999px;
                height: 0.55rem;
                margin: 0.85rem 0 0.2rem;
                overflow: hidden;
                width: 100%;
            }

            .viva-progress-fill {
                background: linear-gradient(90deg, var(--teal), var(--blue));
                height: 100%;
                transition: width 240ms ease;
            }

            .viva-examiner-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-left: 3px solid var(--teal);
                border-radius: 8px;
                box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
                margin: 1rem 0;
                padding: 1.1rem 1.25rem;
            }

            .viva-examiner-label {
                align-items: center;
                color: var(--teal);
                display: flex;
                font-size: 0.78rem;
                font-weight: 800;
                gap: 0.4rem;
                margin-bottom: 0.5rem;
                text-transform: uppercase;
            }

            .viva-examiner-question {
                color: #f8fafc;
                font-size: 1.12rem;
                font-weight: 650;
                line-height: 1.5;
            }

            .viva-eval-card {
                border-radius: 8px;
                margin: 1rem 0 0.4rem;
                overflow: hidden;
            }

            .viva-eval-head {
                align-items: center;
                display: flex;
                justify-content: space-between;
                padding: 0.8rem 1rem;
            }

            .viva-eval-head.strong { background: linear-gradient(135deg, rgba(45, 212, 191, 0.22), rgba(45, 212, 191, 0.08)); }
            .viva-eval-head.mid { background: linear-gradient(135deg, rgba(246, 196, 95, 0.22), rgba(246, 196, 95, 0.08)); }
            .viva-eval-head.weak { background: linear-gradient(135deg, rgba(251, 113, 133, 0.22), rgba(251, 113, 133, 0.08)); }

            .viva-eval-title {
                color: #f8fafc;
                font-weight: 800;
            }

            .viva-score-pill {
                background: rgba(2, 6, 23, 0.35);
                border-radius: 999px;
                color: #f8fafc;
                font-size: 0.86rem;
                font-weight: 850;
                padding: 0.3rem 0.75rem;
            }

            .viva-report-chip {
                background: rgba(2, 6, 23, 0.35);
                border-radius: 999px;
                color: #cbd5e1;
                font-size: 0.74rem;
                font-weight: 750;
                margin-left: 0.5rem;
                padding: 0.2rem 0.55rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_provider_config():
    provider = os.getenv("LLM_PROVIDER", "Ollama").strip()
    provider_key = provider.lower()
    if provider_key == "gemini":
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        missing_secret = not os.getenv("GOOGLE_API_KEY")
    else:
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        missing_secret = False
    return provider, provider_key, model_name, missing_secret


def get_uploaded_document_count():
    return len(list_uploaded_documents())


def get_indexed_chunk_count():
    try:
        return get_db_collection().count()
    except Exception:
        return 0


def format_status(status):
    return {
        "idle": ("Ready", "No questions asked yet"),
        "success": ("Answered", "Last query completed"),
        "error": ("Needs attention", "Last query failed"),
        "reset": ("Reset", "Knowledge base cleared"),
    }.get(status, ("Ready", "Standing by"))


def assistant_mode_details(mode):
    details = {
        "Answer briefly": "Best for quick revision. The assistant replies in a few direct sentences.",
        "Answer in detail": "Best for assignments and deeper understanding. The assistant retrieves more context and explains fully.",
        "Explain like I am a beginner": "Best for hard topics. The assistant uses simple language and explains terms step by step.",
        "Give a summary": "Best for document overview. The assistant compresses the main ideas into a readable summary.",
        "Generate key points": "Best for exam prep. The assistant returns crisp revision bullets.",
    }
    return details.get(mode, "Choose how TutoAI should shape the next answer.")


def render_html(html):
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def default_messages():
    return [
        {
            "role": "assistant",
            "content": "Welcome to TutoAI. Upload documents (PDF, Word, PowerPoint, text, Markdown, RTF, or images/scans), train the knowledge base, then ask a question or try a sample prompt.",
            "sources": [],
            "response_mode": None,
        }
    ]


def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_PATH):
        return {"messages": None, "feedback": {}}

    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"messages": None, "feedback": {}}

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        messages = None

    feedback = data.get("feedback", {})
    if isinstance(feedback, dict):
        feedback = {
            int(key): value
            for key, value in feedback.items()
            if str(key).isdigit() and value in {"Helpful", "Not Helpful"}
        }
    else:
        feedback = {}

    return {"messages": messages, "feedback": feedback}


def save_chat_history(messages=None, feedback=None):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages if messages is not None else st.session_state.get("messages", []),
        "feedback": feedback if feedback is not None else st.session_state.get("feedback", {}),
    }
    try:
        with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
    except (OSError, TypeError):
        st.warning("Chat history could not be saved, but the current conversation is still available.")


def build_project_report(analytics, provider, model_name):
    documents = analytics.get("documents", [])
    doc_lines = [
        f"- {item['name']}: {item['pages']} pages, {item['chunks']} indexed chunks"
        for item in documents
    ]
    topic_lines = [
        f"- {item['topic'].title()}: {item['count']} mentions"
        for item in analytics.get("top_topics", [])[:8]
    ]
    return "\n".join(
        [
            "TutoAI - AI Knowledge Agent Project Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "Objective",
            "TutoAI is a retrieval-augmented research assistant that helps students upload notes in almost any format, build a local knowledge base, ask grounded questions, generate quizzes, summarize documents, compare documents, and inspect the sources used for each answer.",
            "",
            "Technology Stack",
            "- Frontend: Streamlit",
            "- Document parsing: pypdf, python-docx, python-pptx, striprtf (PDF, Word, PowerPoint, RTF)",
            "- OCR: Tesseract via pytesseract, for scanned PDFs and images (JPG/PNG/BMP/TIFF)",
            "- Embeddings: sentence-transformers all-MiniLM-L6-v2",
            "- Vector database: ChromaDB",
            f"- LLM provider: {provider} ({model_name})",
            "",
            "Implemented Features",
            "- Unified multi-format ingestion: PDF, Word (.docx), PowerPoint (.pptx), TXT, Markdown, RTF, and images, with automatic OCR fallback for scanned PDFs and images",
            "- Multi-document upload and persistent saved documents",
            "- RAG question answering with source cards, snippets, page/slide/section badges, relevance indicators, and context preview",
            "- Visual citations: cited PDF pages are rendered with the exact passage highlighted",
            "- Smart assistant modes for brief, detailed, beginner, summary, and key-point answers",
            "- Dashboard analytics with pages/units per document, chunk counts, detected topics, and documents-by-topic summary",
            "- Document summaries and cross-format document comparison",
            "- Gradeable quiz generation (MCQ and descriptive) with per-question logging",
            "- Spaced-repetition review queue (SM-2) for questions answered incorrectly",
            "- Topic mastery dashboard built from quiz and review history",
            "- Session chat history, feedback buttons, and answer export as text/PDF",
            "",
            "Current Knowledge Base",
            *(doc_lines or ["- No documents indexed yet."]),
            "",
            "Detected Topics",
            *(topic_lines or ["- No topics detected yet."]),
            "",
            "Presentation Flow",
            "1. Upload documents and click Ingest & Train.",
            "2. Ask a question in Chat & Research and show the source cards + highlighted citation image.",
            "3. Open Dashboard to show analytics.",
            "4. Generate a document summary or compare two documents.",
            "5. Generate a quiz, answer it, and show the Review Due and Mastery Dashboard tabs update live.",
        ]
    )


def metric_card(label, value, caption, icon, accent):
    st.markdown(
        f"""
        <div class="metric-card {accent}">
            <div class="metric-top">
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-icon">{icon}</div>
            </div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(icon, title):
    st.markdown(
        f'<div class="tuto-section-title"><span>{icon}</span><span>{escape(title)}</span></div>',
        unsafe_allow_html=True,
    )


def mode_badge(mode):
    if mode:
        st.markdown(
            f'<div class="mode-badge">{escape(mode)}</div>',
            unsafe_allow_html=True,
        )


def difficulty_badge_html(level):
    variant = str(level or "").strip().lower()
    if variant not in {"beginner", "intermediate", "advanced"}:
        variant = "intermediate"
    icon = {"beginner": "🌱", "intermediate": "⚡", "advanced": "🔥"}[variant]
    return f'<span class="difficulty-badge {variant}">{icon} {escape(str(level))}</span>'


def viva_progress_html(current, total):
    total = max(total, 1)
    pct = max(0, min(100, round((current / total) * 100)))
    return (
        '<div class="viva-progress-track">'
        f'<div class="viva-progress-fill" style="width: {pct}%"></div>'
        "</div>"
    )


def viva_score_band(score, max_score=10):
    """Classify a viva answer score into strong / mid / weak for card styling."""
    try:
        ratio = float(score) / float(max_score)
    except (TypeError, ValueError, ZeroDivisionError):
        return "mid"
    if ratio >= 0.7:
        return "strong"
    if ratio >= 0.4:
        return "mid"
    return "weak"


def _locator_label(source_name: str) -> str:
    """PDFs get 'Page', PPTX gets 'Slide', everything else gets 'Section' —
    matches the locator rag_engine puts in each source's 'page' field."""
    ext = detect_document_type(source_name)
    if ext == ".pptx":
        return "Slide"
    if ext == ".pdf":
        return "Page"
    return "Section"


def source_cards(sources, message_id):
    if not sources:
        return

    cards = []
    for index, node in enumerate(sources, start=1):
        source_raw = str(node.get("source", "Unknown source"))
        source = escape(source_raw)
        icon = file_type_icon(source_raw)
        locator_label = _locator_label(source_raw)
        page = escape(str(node.get("page", "N/A")))
        snippet = escape(str(node.get("snippet", "No preview available.")))
        relevance = int(node.get("relevance") or 0)
        label = escape(str(node.get("relevance_label", "Unknown")))
        anchor = f"context-{message_id}-{index}"
        cards.append(
            textwrap.dedent(
                f"""
            <a class="source-card-link" href="#{anchor}">
                <div class="source-card-head">
                    <div class="source-file">{icon} {source}</div>
                    <div class="page-badge">{locator_label} {page}</div>
                </div>
                <div class="source-snippet">{snippet}</div>
                <div class="relevance-row">
                    <div class="relevance-badge">{label} {relevance}%</div>
                    <div class="relevance-track"><div class="relevance-fill" style="width: {relevance}%"></div></div>
                </div>
            </a>
            """
            ).strip()
        )

    render_html(f'<div class="source-grid">{"".join(cards)}</div>')


def context_used_section(sources, message_id):
    """Shows the raw retrieved context per source, plus — where possible —
    the actual PDF page rendered as an image with the cited passage
    highlighted (visual citations only apply to PDF sources; other
    formats show text context only)."""
    if not sources:
        return

    with st.expander("Show context used"):
        for index, node in enumerate(sources, start=1):
            source_raw = str(node.get("source", "Unknown source"))
            source = escape(source_raw)
            locator_label = _locator_label(source_raw)
            page_raw = node.get("page", "N/A")
            page = escape(str(page_raw))
            context = escape(str(node.get("context", node.get("snippet", ""))))
            render_html(
                textwrap.dedent(
                    f"""
                <div class="context-block" id="context-{message_id}-{index}">
                    <div class="context-title">{source} - {locator_label} {page}</div>
                    <div class="context-text">{context}</div>
                </div>
                """
                ).strip()
            )

            img_bytes = None
            if isinstance(page_raw, int) and detect_document_type(source_raw) == ".pdf":
                snippet_for_search = node.get("context", node.get("snippet", ""))
                img_bytes = get_citation_image(node.get("source", ""), page_raw, snippet_for_search)

            if img_bytes:
                st.image(
                    img_bytes,
                    caption=f"Highlighted excerpt — {node.get('source', '')}, page {page_raw}",
                    use_container_width=True,
                )


def escape_pdf_text(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf_bytes(title, body):
    lines = [title, "", *body.splitlines()]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=92) or [""])

    pages = [wrapped[i : i + 42] for i in range(0, len(wrapped), 42)] or [[""]]
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    page_refs = []

    for page_index, page_lines in enumerate(pages):
        page_obj = 3 + page_index * 2
        stream_obj = page_obj + 1
        page_refs.append(f"{page_obj} 0 R")
        y = 760
        commands = ["BT", "/F1 10 Tf", "14 TL"]
        for line in page_lines:
            commands.append(f"72 {y} Td ({escape_pdf_text(line)}) Tj")
            y -= 16
        commands.append("ET")
        stream = "\n".join(commands)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f"/Contents {stream_obj} 0 R >>"
        )
        objects.append(f"<< /Length {len(stream.encode('latin-1', 'replace'))} >>\nstream\n{stream}\nendstream")

    objects.insert(1, f"<< /Type /Pages /Kids [{' '.join(page_refs)}] /Count {len(page_refs)} >>")
    pdf = ["%PDF-1.4\n"]
    offsets = [0]
    for obj_num, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part.encode("latin-1", "replace")) for part in pdf))
        pdf.append(f"{obj_num} 0 obj\n{obj}\nendobj\n")
    xref_offset = sum(len(part.encode("latin-1", "replace")) for part in pdf)
    pdf.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.append(f"{offset:010d} 00000 n \n")
    pdf.append(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF")
    return "".join(pdf).encode("latin-1", "replace")


# ---------------------------------------------------------------------------
# Professional Viva PDF Report (ReportLab) -- additive, sits alongside the
# existing plain-text Viva_Report.txt download; does not replace it.
# ---------------------------------------------------------------------------
def _parse_evaluation_sections(evaluation_text):
    """Splits evaluate_viva_answer()'s known "Correct Answer / Strengths /
    Missing Points / Suggestions" format into a dict. Falls back to putting
    everything under 'raw' if the text doesn't match (e.g. an older saved
    evaluation) so report generation never breaks on unexpected format."""

    sections = {"correct_answer": "", "strengths": "", "weaknesses": "", "suggestions": "", "raw": evaluation_text or ""}

    pattern = re.compile(
        r"(Correct Answer|Strengths|Missing Points|Suggestions)\s*:\s*(.*?)"
        r"(?=(?:Correct Answer|Strengths|Missing Points|Suggestions)\s*:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    key_map = {
        "correct answer": "correct_answer",
        "strengths": "strengths",
        "missing points": "weaknesses",
        "suggestions": "suggestions",
    }

    found_any = False
    for label, body in pattern.findall(evaluation_text or ""):
        key = key_map.get(label.strip().lower())
        if key:
            sections[key] = body.strip()
            found_any = True

    if not found_any:
        sections["raw"] = (evaluation_text or "").strip()

    return sections


def _viva_overall_remark(percent):
    if percent >= 85:
        return "Excellent grasp of the material, with consistently strong and well-communicated answers."
    if percent >= 70:
        return "Solid overall understanding, with a few areas that would benefit from reinforcement."
    if percent >= 50:
        return "Foundational understanding is present, but several concepts need further review."
    return "Significant gaps were identified; a focused review of the source material is recommended."


def build_viva_pdf_report(progress, student_name, level, percent):
    """Builds a formatted, multi-section PDF viva report with ReportLab:
    student/date/difficulty header, final score + confidence summary,
    an overall remark, then a per-question breakdown (question, answer,
    evaluation broken into correct-answer/strengths/weaknesses/suggestions,
    and that question's confidence metrics if available)."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="TutoAI Viva Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TutoTitle", parent=styles["Title"], fontSize=20, spaceAfter=4)
    h2_style = ParagraphStyle("TutoH2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#0f172a"))
    h3_style = ParagraphStyle("TutoH3", parent=styles["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#0f172a"))
    body_style = ParagraphStyle("TutoBody", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_LEFT)
    muted_style = ParagraphStyle("TutoMuted", parent=styles["BodyText"], fontSize=9, textColor=colors.HexColor("#475569"))

    story = []

    story.append(Paragraph("TutoAI — Viva Report", title_style))
    story.append(Paragraph("AI Viva Simulator performance record", muted_style))
    story.append(Spacer(1, 10))

    meta_table = Table(
        [
            ["Student Name", student_name or "Anonymous Student"],
            ["Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Difficulty", level],
            ["Questions Answered", str(len(progress.get("history", [])))],
            ["Final Score", str(progress.get("score", 0))],
            ["Percentage", f"{percent}%"],
        ],
        colWidths=[4.5 * cm, 10.5 * cm],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e2e8f0")),
            ]
        )
    )
    story.append(meta_table)

    # Confidence Summary -- averaged across every answer that has a
    # confidence breakdown attached (older histories may not).
    history = progress.get("history", [])
    conf_items = [h["confidence"] for h in history if isinstance(h.get("confidence"), dict)]
    if conf_items:
        def _avg(key):
            values = [item.get(key, 0) for item in conf_items]
            return round(sum(values) / len(values), 1) if values else 0

        story.append(Paragraph("Confidence Summary", h2_style))
        conf_table = Table(
            [
                ["Confidence", "Accuracy", "Completeness", "Communication"],
                [f"{_avg('confidence')}%", f"{_avg('accuracy')}%", f"{_avg('completeness')}%", f"{_avg('communication')}%"],
            ],
            colWidths=[3.75 * cm] * 4,
        )
        conf_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ]
            )
        )
        story.append(conf_table)

    story.append(Paragraph("Overall Remarks", h2_style))
    story.append(Paragraph(escape(_viva_overall_remark(percent)), body_style))

    story.append(Paragraph("Question-by-Question Breakdown", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cbd5e1")))

    for i, item in enumerate(history, start=1):
        sections = _parse_evaluation_sections(item.get("evaluation", ""))
        conf = item.get("confidence") if isinstance(item.get("confidence"), dict) else None

        story.append(Paragraph(f"Question {i}", h3_style))
        story.append(Paragraph(escape(item.get("question", "")), body_style))

        story.append(Paragraph("Answer", muted_style))
        story.append(Paragraph(escape(item.get("answer", "")) or "-", body_style))

        if sections["raw"] and not any([sections["correct_answer"], sections["strengths"], sections["weaknesses"], sections["suggestions"]]):
            story.append(Paragraph("Evaluation", muted_style))
            story.append(Paragraph(escape(sections["raw"]), body_style))
        else:
            if sections["correct_answer"]:
                story.append(Paragraph("Correct Answer", muted_style))
                story.append(Paragraph(escape(sections["correct_answer"]), body_style))
            if sections["strengths"]:
                story.append(Paragraph("Strengths", muted_style))
                story.append(Paragraph(escape(sections["strengths"]), body_style))
            if sections["weaknesses"]:
                story.append(Paragraph("Weaknesses", muted_style))
                story.append(Paragraph(escape(sections["weaknesses"]), body_style))
            if sections["suggestions"]:
                story.append(Paragraph("Suggestions", muted_style))
                story.append(Paragraph(escape(sections["suggestions"]), body_style))

        if conf:
            story.append(
                Paragraph(
                    f"Confidence {conf.get('confidence', 0)}% · Accuracy {conf.get('accuracy', 0)}% · "
                    f"Completeness {conf.get('completeness', 0)}% · Communication {conf.get('communication', 0)}%",
                    muted_style,
                )
            )

        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#e2e8f0")))

    doc.build(story)
    return buffer.getvalue()


def build_answer_export(message):
    sources = message.get("sources", [])
    source_lines = [
        f"- {node.get('source', 'Unknown')} page {node.get('page', 'N/A')} "
        f"({node.get('relevance_label', 'Unknown')} {node.get('relevance', 0)}%)"
        for node in sources
    ]
    return "\n".join(
        [
            "TutoAI Answer Export",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Mode: {message.get('response_mode') or 'N/A'}",
            "",
            "Answer:",
            message.get("content", ""),
            "",
            "Sources:",
            "\n".join(source_lines) if source_lines else "No sources attached.",
        ]
    )


def render_answer_actions(message, index):
    if message.get("role") != "assistant" or not message.get("response_mode"):
        return

    export_text = build_answer_export(message)
    cols = st.columns([1, 1, 1, 1, 4])
    with cols[0]:
        st.download_button(
            "Text",
            export_text,
            file_name=f"tutoai-answer-{index}.txt",
            mime="text/plain",
            key=f"text-export-{index}",
        )
    with cols[1]:
        st.download_button(
            "PDF",
            make_pdf_bytes("TutoAI Answer Export", export_text),
            file_name=f"tutoai-answer-{index}.pdf",
            mime="application/pdf",
            key=f"pdf-export-{index}",
        )
    with cols[2]:
        if st.button("Helpful", key=f"helpful-{index}"):
            st.session_state.feedback[index] = "Helpful"
            save_chat_history()
            st.toast("Feedback saved: Helpful")
    with cols[3]:
        if st.button("Not Helpful", key=f"not-helpful-{index}"):
            st.session_state.feedback[index] = "Not Helpful"
            save_chat_history()
            st.toast("Feedback saved: Not Helpful")


def init_session_state():
    saved_chat = load_chat_history()

    if "messages" not in st.session_state:
        st.session_state.messages = saved_chat["messages"] or default_messages()

    if "index_built" not in st.session_state:
        st.session_state.index_built = has_indexed_data()

    if "last_query_status" not in st.session_state:
        st.session_state.last_query_status = "idle"

    if "response_mode" not in st.session_state:
        st.session_state.response_mode = get_assistant_modes()[0]

    if "response_mode_picker" not in st.session_state:
        st.session_state.response_mode_picker = st.session_state.response_mode

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if "pending_mode" not in st.session_state:
        st.session_state.pending_mode = None

    if "feedback" not in st.session_state:
        st.session_state.feedback = saved_chat["feedback"]


def run_chat_prompt(prompt, selected_mode):
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": [], "response_mode": None}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and preparing a grounded answer..."):
            try:
                response = query_knowledge_base(
                    prompt,
                    selected_mode,
                    chat_history=st.session_state.messages[:-1],
                )
                answer_text = response["response"]
                sources = response.get("source_nodes", [])
                response_mode = response.get("response_mode", selected_mode)

                mode_badge(response_mode)
                st.markdown(answer_text)
                message_id = len(st.session_state.messages)
                source_cards(sources, message_id)
                context_used_section(sources, message_id)

                st.session_state.last_query_status = "success"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "response_mode": response_mode,
                    }
                )
                save_chat_history()
            except Exception as exc:
                err_msg = (
                    "I could not complete that request. Check that your model provider is running "
                    f"and your document index is ready. Details: {exc}"
                )
                st.error(err_msg)
                st.session_state.last_query_status = "error"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                        "response_mode": selected_mode,
                    }
                )
                save_chat_history()

    st.rerun()


inject_styles()
init_session_state()

provider, provider_key, model_name, missing_secret = get_provider_config()

with st.sidebar:
    st.markdown(
        """
        <div class="tuto-sidebar-brand">
            <div class="brand-title">TutoAI</div>
            <div class="brand-caption">Project-ready multi-format research assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_title("⚙", "Configuration")
    if missing_secret:
        st.error("Gemini is selected, but GOOGLE_API_KEY is missing in .env.")
    else:
        st.success(f"{provider} connected")
    st.caption(f"Model: {model_name}")

    st.divider()
    render_section_title("📚", "Knowledge Base")
    saved_pdfs = list_uploaded_documents()
    if saved_pdfs:
      st.caption(f"Saved documents available: {len(saved_pdfs)}")

    with st.expander("View saved documents"):

        for pdf_name in saved_pdfs:

            col1, col2 = st.columns([8, 1])

            with col1:
                st.write(f"{file_type_icon(pdf_name)} {pdf_name}")

            with col2:
                if st.button("🗑", key=f"delete_{pdf_name}"):

                    rag_engine.delete_document(pdf_name)

                    st.success(f"{pdf_name} deleted successfully!")

                    st.session_state.index_built = has_indexed_data()

                    st.rerun()
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, Word, PowerPoint, TXT, MD, RTF, images/scans)",
        type=UPLOAD_EXTENSIONS,
        accept_multiple_files=True,
    )

    if st.button("Ingest & Train", type="primary", use_container_width=True):
        if uploaded_files or saved_pdfs:
            with st.spinner(
                "Extracting text (with OCR where needed), chunking, and building the vector index..."
            ):
                try:
                    for uploaded_file in uploaded_files:
                        save_uploaded_file(uploaded_file)
                    status = build_index()
                    st.session_state.index_built = has_indexed_data()
                    st.session_state.last_query_status = "idle"
                    st.success(status)
                    st.rerun()
                except Exception as exc:
                    st.session_state.last_query_status = "error"
                    st.error(
                        "The upload was saved, but indexing failed. Try a smaller file or check "
                        f"your embedding model setup. Details: {exc}"
                    )
        else:
            st.warning("Upload a document first, or keep a previously saved document in the knowledge base.")

    if st.button("Reset Knowledge Base", use_container_width=True):
        clear_data()
        st.session_state.index_built = False
        st.session_state.last_query_status = "reset"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Knowledge base was reset. Upload new documents to get started.",
                "sources": [],
                "response_mode": None,
            }
        ]
        st.session_state.feedback = {}
        # Clear any in-progress quiz state tied to the old knowledge base
        for key in [
            "quiz_questions", "quiz_type", "quiz_topic",
            "quiz_graded", "quiz_user_answers", "quiz_provider",
        ]:
            st.session_state.pop(key, None)
        save_chat_history()
        st.success("Reset complete.")
        st.rerun()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat history cleared. Ask a fresh question whenever you are ready.",
                "sources": [],
                "response_mode": None,
            }
        ]
        st.session_state.feedback = {}
        save_chat_history()
        st.rerun()

    st.divider()
    if has_indexed_data():
        st.success("Knowledge base is ready")
    elif saved_pdfs:
        st.info("Saved documents found. Click Ingest & Train to use them again.")
    else:
        st.warning("No documents indexed yet")

    due_count = progress_store.get_due_count()
    if due_count:
        st.info(f"🔁 {due_count} review question(s) due — see the Review Due tab")


pdf_count = get_uploaded_document_count()
chunk_count = get_indexed_chunk_count()
status_value, status_caption = format_status(st.session_state.last_query_status)
index_ready = chunk_count > 0

st.markdown(
    f"""
    <section class="tuto-hero">
        <div class="tuto-kicker">AI Knowledge Agent</div>
        <h1>TutoAI</h1>
        <p>
            A project-ready research dashboard for turning uploaded documents into sourced answers,
            document analytics, summaries, comparisons, and practice questions. Supports PDF, Word,
            PowerPoint, text, Markdown, RTF, images, and scanned documents via OCR.
        </p>
        <div class="tuto-status-pill">
            <span>{"●" if index_ready else "○"}</span>
            <span>{"Knowledge base ready" if index_ready else "Upload documents to activate chat"}</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

card_cols = st.columns(4)
with card_cols[0]:
    metric_card("Total Uploaded Documents", pdf_count, "Documents stored in KB", "📄", "teal")
with card_cols[1]:
    metric_card("Indexed Chunks", chunk_count, "Searchable vector records", "▦", "amber")
with card_cols[2]:
    metric_card("Active Provider", provider, model_name, "⚡", "blue")
with card_cols[3]:
    metric_card("Last Query Status", status_value, status_caption, "✓", "rose")

if not index_ready:
    if pdf_count:
        st.info(
            "Saved documents are still available. Click Ingest & Train in the sidebar to rebuild "
            "the index and continue without uploading again."
        )
    else:
        st.info(
            "Start here: upload documents in the sidebar, click Ingest & Train, then try a sample "
            "prompt or open the Guide tab for presentation instructions."
        )

tab_chat, tab_dashboard, tab_docs, tab_learning, tab_quiz,tab_viva, tab_review, tab_mastery, tab_viva_analytics, tab_study_recs, tab_guide = st.tabs(
    [
        "Chat & Research",
        "Dashboard",
        "Document Tools",
        "Learning Path",
        "Quiz Generator",
        "AI Viva",
        "Review Due",
        "Mastery Dashboard",
        "Viva Analytics",
        "AI Study Recommendations",
        "Guide",
    ]
)


with tab_chat:
    render_section_title("💬", "Chat & Research")
    render_section_title("✨", "Assistant Workflow")
    assistant_modes = get_assistant_modes()

    if st.session_state.pending_mode:
        st.session_state.response_mode_picker = st.session_state.pending_mode

    selected_mode = st.pills(
        "Assistant workflow",
        assistant_modes,
        selection_mode="single",
        key="response_mode_picker",
        label_visibility="collapsed",
        disabled=not index_ready,
    )
    selected_mode = selected_mode or st.session_state.response_mode or assistant_modes[0]
    st.session_state.response_mode = selected_mode
    st.info(f"Current mode: **{selected_mode}**. {assistant_mode_details(selected_mode)}")

    render_section_title("💡", "Sample Prompts")
    sample_cols = st.columns(4)
    sample_prompts = [
        ("Summarize the most important ideas in these documents.", "Give a summary"),
        ("Generate key points I should revise for an exam.", "Generate key points"),
        ("Explain this topic like I am a beginner.", "Explain like I am a beginner"),
        ("What is the difference between these files?", "Answer in detail"),
    ]
    for col, (sample, sample_mode) in zip(sample_cols, sample_prompts):
        with col:
            if st.button(sample, disabled=not index_ready, use_container_width=True):
                st.session_state.pending_prompt = sample
                st.session_state.pending_mode = sample_mode
                st.rerun()

    for index, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                mode_badge(msg.get("response_mode"))
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                source_cards(msg.get("sources", []), index)
                context_used_section(msg.get("sources", []), index)
                render_answer_actions(msg, index)

    if not index_ready:
        st.info("Upload and ingest at least one document before asking document questions.")

    typed_prompt = st.chat_input(
        "Ask a question, or ask a follow-up like 'explain the second point more'...",
        disabled=not index_ready,
    )
    prompt_to_run = st.session_state.pending_prompt or typed_prompt
    mode_to_run = st.session_state.pending_mode or selected_mode
    if st.session_state.pending_prompt:
        st.session_state.pending_prompt = None
        st.session_state.pending_mode = None
    run_chat_prompt(prompt_to_run, mode_to_run)


with tab_dashboard:
    render_section_title("📊", "Document Analytics")
    analytics = get_document_analytics()

    analytics_cols = st.columns(3)
    with analytics_cols[0]:
        metric_card("Total Pages/Units", analytics["total_pages"], "Across uploaded documents", "☷", "teal")
    with analytics_cols[1]:
        metric_card("Total Chunks", analytics["total_chunks"], "Indexed for retrieval", "▦", "amber")
    with analytics_cols[2]:
        metric_card("Detected Topics", len(analytics["top_topics"]), "From indexed text", "◎", "blue")

    render_section_title("📄", "Pages Per Uploaded Document")
    if analytics["documents"]:
        st.dataframe(
            [
                {
                    "Document": f"{file_type_icon(item['name'])} {item['name']}",
                    "Type": item["file_type"].upper(),
                    "Pages/Units": item["pages"],
                    "Indexed": item["indexed_pages"],
                    "Chunks": item["chunks"],
                    "Top Topics": ", ".join(topic["topic"] for topic in item["topics"][:4]),
                }
                for item in analytics["documents"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Upload and index documents to view document analytics.")

    render_section_title("🔎", "Top Relevant Topics Detected")
    topic_cols = st.columns(4)
    for idx, topic in enumerate(analytics["top_topics"][:8]):
        with topic_cols[idx % 4]:
            render_html(
                textwrap.dedent(
                    f"""
                <div class="topic-card">
                    <div class="topic-title">{escape(topic["topic"])}</div>
                    <p class="topic-meta">{topic["count"]} indexed mentions</p>
                </div>
                """
                ).strip()
            )

    render_section_title("🧭", "Documents By Topic")
    if analytics["documents_by_topic"]:
        for topic_info in analytics["documents_by_topic"]:
            docs = topic_info["documents"]
            if docs:
                st.markdown(
                    f"**{topic_info['topic'].title()}**: "
                    + ", ".join(f"{item['document']} ({item['mentions']})" for item in docs)
                )
    else:
        st.caption("No topic summary available yet.")


with tab_docs:
    render_section_title("🧰", "Document Tools")
    pdf_names = list_uploaded_documents()

    summary_col, compare_col = st.columns(2)
    with summary_col:
        st.markdown("#### Document Summary")
        summary_target = st.selectbox(
            "Choose a document to summarize",
            ["All indexed documents", *pdf_names],
            disabled=not index_ready,
        )
        if st.button("Generate Summary", type="primary", disabled=not index_ready):
            target = None if summary_target == "All indexed documents" else summary_target
            with st.spinner("Creating a study-ready summary..."):
                summary = generate_document_summary(target)
                st.session_state.latest_summary = summary
        if st.session_state.get("latest_summary"):
            st.markdown(st.session_state.latest_summary)
            st.download_button(
                "Export Summary",
                st.session_state.latest_summary,
                file_name="tutoai-document-summary.txt",
                mime="text/plain",
            )

    with compare_col:
        st.markdown("#### Compare Two Documents")
        left = st.selectbox("First document", pdf_names, disabled=not index_ready or len(pdf_names) < 2)
        right = st.selectbox(
            "Second document",
            pdf_names,
            index=1 if len(pdf_names) > 1 else 0,
            disabled=not index_ready or len(pdf_names) < 2,
        )
        if st.button("Compare Documents", type="primary", disabled=not index_ready or len(pdf_names) < 2):
            with st.spinner("Comparing both documents against their indexed content..."):
                st.session_state.latest_comparison = compare_documents(left, right)
        if st.session_state.get("latest_comparison"):
            st.markdown(st.session_state.latest_comparison)
            st.download_button(
                "Export Comparison",
                st.session_state.latest_comparison,
                file_name="tutoai-document-comparison.txt",
                mime="text/plain",
            )

with tab_learning:

    render_section_title("🛣️", "AI Learning Path Generator")

    topic = st.text_input("Topic")

    level = st.selectbox(
        "Current Level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    goal = st.text_input("Learning Goal")

    duration = st.slider(
        "Duration (weeks)",
        1,
        12,
        4
    )

    # -----------------------------
    # Generate Learning Path
    # -----------------------------
    if st.button("Generate Learning Path", use_container_width=True):

        try:

            roadmap = generate_learning_path(
                topic,
                level,
                goal,
                duration
            )

            roadmap = json.loads(roadmap)

            # Save roadmap permanently
            st.session_state.learning_path = roadmap

            # Reset old checkbox values
            for i in range(len(roadmap)):
                st.session_state[f"day_{i}"] = False

        except json.JSONDecodeError:
            st.error("❌ The AI returned an invalid learning path format.")
            st.stop()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    # ----------------------------------------------------
    # Display roadmap if already generated
    # ----------------------------------------------------
    if "learning_path" in st.session_state:

        roadmap = st.session_state.learning_path

        st.success("🎯 Personalized Learning Path Generated")

        progress = 0

        total_time = sum(
            int(item["time"].split()[0])
            for item in roadmap
            if item["time"].split()[0].isdigit()
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("📚 Total Topics", len(roadmap))

        with col2:
            st.metric("⏱ Total Study Time", f"{total_time} min")

        st.divider()

        # -----------------------------
        # Roadmap Cards
        # -----------------------------
        for i, step in enumerate(roadmap):

            completed = st.checkbox(
                f"Mark Day {i+1} as Completed",
                key=f"day_{i}"
            )

            if completed:
                progress += 1

            st.markdown(
                f"""
<div style="
padding:20px;
border-radius:15px;
background:#1E293B;
border-left:6px solid #38BDF8;
margin-bottom:18px;
box-shadow:0 2px 8px rgba(0,0,0,.25);
">

<h4 style="margin:0;color:#38BDF8;">
📅 Day {i+1}
</h4>

<h3 style="margin-top:10px;">
{step["topic"]}
</h3>

<p style="font-size:16px;">
⏱ Estimated Time:
<b>{step["time"]}</b>
</p>

</div>
""",
                unsafe_allow_html=True
            )

        # -----------------------------
        # Progress
        # -----------------------------
        percent = int((progress / len(roadmap)) * 100)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "✅ Completed",
                f"{progress}/{len(roadmap)}"
            )

        with col2:
            st.metric(
                "📈 Overall Progress",
                f"{percent}%"
            )

        st.progress(percent / 100)

        # -----------------------------
        # Download Button
        # -----------------------------
        download_text = ""

        for i, step in enumerate(roadmap):

            status = "Completed" if st.session_state[f"day_{i}"] else "Pending"

            download_text += (
                f"Day {i+1}\n"
                f"Topic : {step['topic']}\n"
                f"Estimated Time : {step['time']}\n"
                f"Status : {status}\n\n"
            )

        st.download_button(
            label="📥 Download Learning Path",
            data=download_text,
            file_name="Learning_Path.txt",
            mime="text/plain",
            use_container_width=True
        )
with tab_quiz:
    render_section_title("📝", "Quiz Generator")
    st.caption(
        "Every question you answer here is saved as a spaced-repetition card — "
        "wrong answers resurface in the Review Due tab, and your accuracy per topic "
        "shows up on the Mastery Dashboard."
    )
    if not index_ready:
        st.info("Upload and ingest at least one document before generating a quiz.")

    col1, col2 = st.columns([1.4, 1])
    with col1:
        topic = st.text_input(
            "Enter topic for questions",
            placeholder="e.g., Photosynthesis, Chapter 1",
            disabled=not index_ready,
        )
    with col2:
        q_type = st.selectbox("Question type", ["MCQ", "Descriptive"], disabled=not index_ready)
        num_q = st.slider("Number of questions", 1, 20, 10, disabled=not index_ready)

    if st.button("Generate Questions", type="primary", disabled=not index_ready):
        if not topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner(f"Generating {q_type}s for '{topic}'..."):
                try:
                    questions, provider_used, message = generate_quiz(topic, num_q, q_type)
                    if not questions:
                        st.warning(message or "Couldn't generate questions for this topic.")
                        st.session_state.pop("quiz_questions", None)
                    else:
                        st.session_state.quiz_questions = questions
                        st.session_state.quiz_type = q_type
                        st.session_state.quiz_topic = topic
                        st.session_state.quiz_graded = False
                        st.session_state.quiz_provider = provider_used
                        st.session_state.pop("quiz_user_answers", None)
                except Exception as exc:
                    st.error(f"Quiz generation failed. Details: {exc}")

    if st.session_state.get("quiz_questions"):
        questions = st.session_state.quiz_questions
        qtype = st.session_state.quiz_type

        st.markdown("### Answer the Questions")

        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}. {q.get('question', '')}**")
                if qtype == "MCQ":
                    st.radio(
                        f"Select answer for Q{i+1}",
                        options=q.get("options", []),
                        key=f"quiz_ans_{i}",
                        label_visibility="collapsed",
                    )
                else:
                    st.text_area(
                        f"Your answer for Q{i+1}", key=f"quiz_ans_{i}", label_visibility="collapsed"
                    )
                st.divider()
            submitted = st.form_submit_button("Submit Answers")

        if submitted:
            st.session_state.quiz_graded = True
            st.session_state.quiz_user_answers = {
                i: st.session_state.get(f"quiz_ans_{i}") for i in range(len(questions))
            }

        if st.session_state.get("quiz_graded"):
            st.markdown("### Results")
            score = 0
            for i, q in enumerate(questions):
                user_ans = st.session_state.get("quiz_user_answers", {}).get(i)

                if qtype == "MCQ":
                    correct_letter = (q.get("correct_answer") or "").strip().upper()[:1]
                    is_correct = bool(user_ans) and user_ans.strip().upper().startswith(correct_letter)
                    if is_correct:
                        score += 1
                        st.success(f"Q{i+1}: Correct ✅")
                    else:
                        st.error(f"Q{i+1}: Incorrect ❌ — Correct answer: {correct_letter}")
                    if q.get("explanation"):
                        st.caption(q["explanation"])
                    if q.get("card_id") and f"logged_{i}" not in st.session_state:
                        progress_store.record_attempt(q["card_id"], is_correct)
                        st.session_state[f"logged_{i}"] = True
                else:
                    st.info(f"Q{i+1}: Model answer — {q.get('model_answer', '')}")
                    st.checkbox(f"I got Q{i+1} right", key=f"self_grade_{i}")

            if qtype == "MCQ":
                st.metric("Score", f"{score} / {len(questions)}")
            else:
                st.caption("Tick the boxes above honestly, then click below to log your results.")
                if st.button("Log Descriptive Results"):
                    for i, q in enumerate(questions):
                        self_correct = st.session_state.get(f"self_grade_{i}", False)
                        if q.get("card_id"):
                            progress_store.record_attempt(q["card_id"], self_correct)
                    st.success("Logged! Check the Mastery Dashboard tab.")

            st.download_button(
                "Export Quiz",
                json.dumps(questions, indent=2, ensure_ascii=False),
                file_name="tutoai-quiz.json",
                mime="application/json",
            )

with tab_viva:

    render_html(
        textwrap.dedent(
            """
        <div class="tuto-hero">
            <div class="tuto-kicker">🎓 Viva Voce</div>
            <h1>AI Examiner</h1>
            <p>A real oral exam, grounded only in your uploaded documents. Answer by voice or
            text — the examiner asks, listens, and scores every response.</p>
        </div>
        """
        ).strip()
    )

    render_section_title("⚙️", "Set Up Your Viva")

    with st.container(border=True):

        setup_col1, setup_col2 = st.columns(2)

        with setup_col1:
            level = st.selectbox(
                "Difficulty",
                ["Beginner", "Intermediate", "Advanced"],
                key="viva_level"
            )

        with setup_col2:
            count = st.slider(
                "Number of Questions",
                1,
                10,
                5,
                key="viva_count"
            )

        topic = st.text_input(
            "Topic (optional)",
            key="viva_topic",
            placeholder="e.g. Retrieval-Augmented Generation, Chapter 2 -- leave blank to cover the whole document set",
        )

        student_name = st.text_input(
            "Student Name (for the report)",
            key="viva_student_name",
            placeholder="e.g. Priya Sharma"
        )

        if st.button(
            "Generate Viva",
            type="primary",
            use_container_width=True
        ):

            try:

                try:
                    questions = generate_viva_questions(
                        level,
                        count,
                        topic
                    )
                except TypeError:
                    # Safety net: falls back to the older 2-argument call
                    # if a stale/older rag_engine1.py (without the topic
                    # parameter) is still loaded, instead of crashing.
                    questions = generate_viva_questions(
                        level,
                        count
                    )

                progress = {
                    "questions": questions,
                    "current": 0,
                    "score": 0,
                    "history": [],
                    "completed": False
                }

                save_progress(progress)

                if "voice_answer" in st.session_state:
                    del st.session_state["voice_answer"]

                st.success("✅ Viva generated successfully.")

                st.rerun()

            except Exception as e:

                st.error(f"Failed to generate viva.\n\n{e}")

    progress = load_progress()

    if not progress["questions"]:

        st.info(
            "Generate a viva to begin."
        )

    elif not progress["completed"]:

        question = current_question()

        total_questions = len(progress["questions"])
        current_index = progress["current"]

        render_html(
            textwrap.dedent(
                f"""
            <div class="viva-badge-row">
                {difficulty_badge_html(st.session_state.get("viva_level", "Intermediate"))}
                <span class="viva-counter-badge">Question {current_index + 1} of {total_questions}</span>
            </div>
            {viva_progress_html(current_index, total_questions)}
            """
            ).strip()
        )

        render_html(
            textwrap.dedent(
                f"""
            <div class="viva-examiner-card">
                <div class="viva-examiner-label">🧑‍🏫 Examiner asks</div>
                <div class="viva-examiner-question">{escape(question["question"])}</div>
            </div>
            """
            ).strip()
        )

        with st.container(border=True):

            st.caption("🎧 Voice Controls")

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "🔊 Speak Question",
                    use_container_width=True
                ):

                    try:
                        audio_bytes = speak(question["question"])

                        if audio_bytes:
                            st.audio(
                                audio_bytes,
                                format="audio/mp3",
                                autoplay=True
                            )
                        else:
                            st.warning("Nothing to speak.")

                    except Exception as e:

                        st.error(f"Speech generation failed.\n\n{e}")

            with col2:

                st.caption("🎙 Record Answer")

                voice_audio = mic_recorder(
                    start_prompt="⏺️ Start Recording",
                    stop_prompt="⏹️ Stop Recording",
                    just_once=True,
                    use_container_width=True,
                    format="webm",
                    key=f"viva_mic_{progress['current']}",
                )

                if voice_audio and voice_audio.get("bytes"):

                    try:
                        text = listen(voice_audio["bytes"])

                        if text:
                            st.session_state["voice_answer"] = text
                            st.rerun()
                        else:
                            st.warning(
                                "Couldn't make out any speech in that recording. "
                                "Try again, or just type your answer below."
                            )

                    except Exception as e:

                        st.error(f"Transcription failed.\n\n{e}")

        with st.container(border=True):

            st.caption("✍️ Your Answer")

            answer = st.text_area(
                "Your Answer",
                value=st.session_state.get(
                    "voice_answer",
                    ""
                ),
                height=180,
                label_visibility="collapsed"
            )

            if st.button(
                "Submit Answer",
                type="primary",
                use_container_width=True
            ):

                if not answer.strip():

                    st.warning("Please answer the question first.")

                else:

                    try:

                        feedback = evaluate_viva_answer(
                            question["question"],
                            answer
                        )

                        score = extract_score(feedback)

                        add_score(score)

                        add_feedback(
                            question["question"],
                            answer,
                            feedback
                        )

                        try:
                            confidence = analyze_viva_confidence(
                                question["question"],
                                answer,
                                feedback
                            )
                        except Exception:
                            confidence = {
                                "confidence": 0,
                                "accuracy": 0,
                                "completeness": 0,
                                "communication": 0
                            }

                        add_confidence(confidence)

                        try:
                            progress_store.add_viva_attempt(
                                level,
                                question["question"],
                                answer,
                                feedback,
                                score,
                                confidence.get("confidence", 0),
                                confidence.get("accuracy", 0),
                                confidence.get("completeness", 0),
                                confidence.get("communication", 0),
                            )
                        except Exception:
                            pass

                        band = viva_score_band(score)

                        render_html(
                            textwrap.dedent(
                                f"""
                            <div class="viva-eval-card">
                                <div class="viva-eval-head {band}">
                                    <span class="viva-eval-title">📝 Examiner's Evaluation</span>
                                    <span class="viva-score-pill">{escape(str(score))} / 10</span>
                                </div>
                            </div>
                            """
                            ).strip()
                        )

                        with st.container(border=True):
                            st.write(feedback)

                        st.markdown("##### 🎯 Confidence Analysis")
                        conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)
                        with conf_col1:
                            st.metric("Confidence", f"{confidence.get('confidence', 0)}%")
                        with conf_col2:
                            st.metric("Accuracy", f"{confidence.get('accuracy', 0)}%")
                        with conf_col3:
                            st.metric("Completeness", f"{confidence.get('completeness', 0)}%")
                        with conf_col4:
                            st.metric("Communication", f"{confidence.get('communication', 0)}%")

                        st.session_state.pop(
                            "voice_answer",
                            None
                        )

                        next_question()

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Evaluation failed.\n\n{e}"
                        )
    # =====================================================
    # Viva Completed
    # =====================================================
    elif progress["completed"]:

        percent = viva_percentage()

        render_html(
            textwrap.dedent(
                f"""
            <div class="tuto-hero">
                <div class="tuto-kicker">🎉 Viva Completed</div>
                <h1>Well Examined!</h1>
                <p>You answered {len(progress["history"])} question(s) at
                {difficulty_badge_html(st.session_state.get("viva_level", "Intermediate"))} level.
                Here's how you did.</p>
            </div>
            """
            ).strip()
        )

        col1, col2 = st.columns(2)

        with col1:
            metric_card(
                "Final Score",
                progress["score"],
                "Total points across all questions",
                "🏆",
                "teal"
            )

        with col2:
            metric_card(
                "Percentage",
                f"{percent}%",
                "Overall viva performance",
                "📈",
                "blue"
            )

        render_html(viva_progress_html(percent, 100))

        conf_items = [
            item["confidence"]
            for item in progress["history"]
            if isinstance(item.get("confidence"), dict)
        ]

        if conf_items:

            def _avg_conf(key):
                values = [item.get(key, 0) for item in conf_items]
                return round(sum(values) / len(values), 1) if values else 0

            st.divider()

            render_section_title("🎯", "Confidence Summary")

            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                metric_card("Confidence", f"{_avg_conf('confidence')}%", "Assertiveness of answers", "💪", "teal")
            with cc2:
                metric_card("Accuracy", f"{_avg_conf('accuracy')}%", "Factual correctness", "🎯", "blue")
            with cc3:
                metric_card("Completeness", f"{_avg_conf('completeness')}%", "Coverage of expected points", "📚", "amber")
            with cc4:
                metric_card("Communication", f"{_avg_conf('communication')}%", "Clarity of expression", "🗣️", "rose")

        st.divider()

        render_section_title("📝", "Viva Report")

        report = ""

        for i, item in enumerate(progress["history"]):

            try:
                item_score = extract_score(item["evaluation"])
                score_chip = f'<span class="viva-report-chip">{escape(str(item_score))} / 10</span>'
            except Exception:
                score_chip = ""

            render_html(
                f'<div class="viva-badge-row"><strong>Question {i + 1}</strong>{score_chip}</div>'
            )

            with st.expander(f"Question {i+1}"):

                st.markdown(
                    f"**Question**\n\n{item['question']}"
                )

                st.markdown(
                    f"**Your Answer**\n\n{item['answer']}"
                )

                st.markdown(
                    f"**Evaluation**\n\n{item['evaluation']}"
                )

            report += (
                f"Question {i+1}\n"
                f"{item['question']}\n\n"
                f"Answer\n"
                f"{item['answer']}\n\n"
                f"Evaluation\n"
                f"{item['evaluation']}\n\n"
                f"{'-'*60}\n\n"
            )

        report_col1, report_col2 = st.columns(2)

        with report_col1:
            st.download_button(
                label="📥 Download Viva Report (TXT)",
                data=report,
                file_name="Viva_Report.txt",
                mime="text/plain",
                use_container_width=True
            )

        with report_col2:
            try:
                pdf_bytes = build_viva_pdf_report(
                    progress,
                    st.session_state.get("viva_student_name", ""),
                    st.session_state.get("viva_level", "Intermediate"),
                    percent,
                )
                st.download_button(
                    label="📄 Download Professional Report (PDF)",
                    data=pdf_bytes,
                    file_name="Viva_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF report generation failed.\n\n{e}")

        st.divider()

        if st.button(
            "🔄 Start New Viva",
            use_container_width=True
        ):

            reset_progress()

            st.session_state.pop(
                "voice_answer",
                None
            )

            st.rerun()

with tab_review:
    render_section_title("🔁", "Questions Due for Review")
    st.caption("Powered by the SM-2 spaced-repetition algorithm — questions you got wrong resurface sooner.")

    due_cards = progress_store.get_due_cards(limit=25)

    if not due_cards:
        st.success("Nothing due right now. Generate a quiz in the Quiz Generator tab, or check back later.")
    else:
        st.info(f"{len(due_cards)} question(s) due for review.")
        for card in due_cards:
            with st.expander(f"[{card['topic']}] {card['question'][:90]}"):
                st.write(card["question"])

                if card["question_type"] == "MCQ":
                    options = json.loads(card["options"]) if card["options"] else []
                    ans = st.radio("Your answer", options, key=f"due_ans_{card['id']}")
                    if st.button("Check Answer", key=f"due_check_{card['id']}"):
                        correct_letter = (card["correct_answer"] or "").strip().upper()[:1]
                        is_correct = bool(ans) and ans.strip().upper().startswith(correct_letter)
                        if is_correct:
                            st.success("Correct! ✅")
                        else:
                            st.error(f"Incorrect ❌ — Correct answer: {correct_letter}")
                        if card["explanation"]:
                            st.caption(card["explanation"])
                        progress_store.record_attempt(card["id"], is_correct)
                else:
                    st.text_area("Your answer", key=f"due_ans_{card['id']}")
                    st.info(f"Model answer: {card['correct_answer']}")
                    self_correct = st.checkbox("I got this right", key=f"due_self_{card['id']}")
                    if st.button("Log Result", key=f"due_log_{card['id']}"):
                        progress_store.record_attempt(card["id"], self_correct)
                        st.success("Logged.")

    st.divider()

    render_section_title("🎤", "Viva Review Due")
    st.caption("Based on your average viva score across every viva you've taken.")

    viva_review = progress_store.get_viva_review_status()

    if viva_review is None:
        st.info("No viva taken yet. Complete a viva in the AI Viva Simulator tab to see a review reminder here.")
    else:
        urgency = viva_review["urgency"]

        review_col1, review_col2 = st.columns(2)
        with review_col1:
            metric_card(
                "Average Viva Score",
                f"{viva_review['average_percent']}%",
                "Across all vivas taken",
                "🎤",
                "blue"
            )
        with review_col2:
            metric_card(
                "Review Status",
                viva_review["label"],
                f"Due {viva_review['review_date']}" if viva_review["review_date"] else "You're on top of this material",
                "⏰",
                "rose" if urgency == "high" else "amber" if urgency == "medium" else "teal"
            )

        if urgency == "high":
            st.error(f"🔴 {viva_review['label']} — your average score is below 40%. Revisit the source material soon.")
        elif urgency == "medium":
            st.warning(f"🟠 {viva_review['label']} — a bit more review will solidify these topics.")
        elif urgency == "low":
            st.info(f"🟡 {viva_review['label']} — a light refresh next week will help retention.")
        else:
            st.success("🟢 No review needed — excellent, consistent viva performance!")


with tab_mastery:
    render_section_title("📊", "Topic Mastery Dashboard")
    st.caption("Accuracy per topic, based on every quiz and review attempt you've logged.")

    mastery = progress_store.get_topic_mastery()

    if not mastery:
        st.info(
            "No quiz attempts recorded yet. Complete a quiz (Quiz Generator tab) or a review "
            "(Review Due tab) to see your breakdown here."
        )
    else:
        df = pd.DataFrame(mastery).sort_values("accuracy")
        st.bar_chart(df.set_index("topic")["accuracy"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        weak_topics = df[df["accuracy"] < 60]
        if not weak_topics.empty:
            st.warning("⚠️ Weak topics (below 60% accuracy): " + ", ".join(weak_topics["topic"].tolist()))
        else:
            st.success("You're above 60% accuracy on every topic attempted so far. Nice work.")

    # =====================================================
    # Overall Mastery -- blends quiz accuracy + viva performance
    # =====================================================
    st.divider()
    render_section_title("🎓", "Overall Mastery")

    dashboard_summary = progress_store.get_dashboard_summary()
    viva_summary = progress_store.get_viva_summary()

    quiz_accuracy = dashboard_summary["overall_accuracy"]
    viva_avg_percent = round((viva_summary["average_score"] / 10.0) * 100, 1) if viva_summary["questions_answered"] else None

    mastery_components = [v for v in (quiz_accuracy if dashboard_summary["total_attempts"] else None, viva_avg_percent) if v is not None]
    overall_mastery = round(sum(mastery_components) / len(mastery_components), 1) if mastery_components else 0.0

    learning_path_roadmap = st.session_state.get("learning_path")
    if learning_path_roadmap:
        completed_days = sum(
            1 for i in range(len(learning_path_roadmap)) if st.session_state.get(f"day_{i}")
        )
        learning_path_percent = round((completed_days / len(learning_path_roadmap)) * 100, 1)
        learning_path_caption = f"{completed_days}/{len(learning_path_roadmap)} days completed"
    else:
        learning_path_percent = None
        learning_path_caption = "No learning path generated yet"

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        metric_card(
            "Overall Mastery %",
            f"{overall_mastery}%",
            "Blended quiz + viva performance",
            "🎓",
            "teal"
        )
    with mc2:
        metric_card(
            "Quiz Accuracy",
            f"{quiz_accuracy}%" if dashboard_summary["total_attempts"] else "—",
            f"{dashboard_summary['total_attempts']} quiz/review attempts logged",
            "📝",
            "blue"
        )
    with mc3:
        metric_card(
            "Average Viva Score",
            f"{viva_avg_percent}%" if viva_avg_percent is not None else "—",
            f"{viva_summary['questions_answered']} viva question(s) answered",
            "🎤",
            "amber"
        )
    with mc4:
        metric_card(
            "Learning Path Completion",
            f"{learning_path_percent}%" if learning_path_percent is not None else "—",
            learning_path_caption,
            "🛣️",
            "rose"
        )

    # =====================================================
    # Topics Studied / Strong / Weak
    # =====================================================
    render_section_title("🏷️", "Topics")

    topics_col1, topics_col2, topics_col3 = st.columns(3)
    with topics_col1:
        metric_card(
            "Topics Studied",
            len(mastery),
            "Distinct topics with quiz/review attempts",
            "📚",
            "blue"
        )
    with topics_col2:
        strong_list = dashboard_summary["strong_topics"]
        st.markdown("**💪 Strong Topics**")
        if strong_list:
            st.success(", ".join(strong_list))
        else:
            st.caption("None yet — keep practicing to build strong topics (≥75% accuracy, 3+ attempts).")
    with topics_col3:
        weak_list = dashboard_summary["weak_topics"]
        st.markdown("**⚠️ Weak Topics**")
        if weak_list:
            st.warning(", ".join(weak_list))
        else:
            st.caption("None — no topics currently below 60% accuracy.")

    # =====================================================
    # Recent Activity
    # =====================================================
    render_section_title("🕒", "Recent Activity")

    activity_col1, activity_col2 = st.columns(2)
    with activity_col1:
        last_viva = viva_summary["last_viva_date"]
        if last_viva:
            try:
                last_viva_display = datetime.fromisoformat(last_viva).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                last_viva_display = last_viva
            st.info(f"🎤 Last Viva Date: **{last_viva_display}**")
        else:
            st.caption("🎤 No viva completed yet.")

    with activity_col2:
        recent_reviews = dashboard_summary["review_history"][-1:] if dashboard_summary["review_history"] else []
        if recent_reviews:
            last_review = recent_reviews[0]
            st.info(
                f"🔁 Last Review Activity: **{last_review['day']}** "
                f"({last_review['correct']}/{last_review['total']} correct)"
            )
        else:
            st.caption("🔁 No review activity logged yet.")


with tab_viva_analytics:
    render_section_title("📈", "Viva Analytics")
    st.caption(
        "Cross-session viva performance, aggregated from every graded answer you've ever "
        "submitted in the AI Viva tab."
    )

    viva_summary_analytics = progress_store.get_viva_summary()
    difficulty_breakdown = progress_store.get_viva_difficulty_breakdown()
    recent_viva_attempts = progress_store.get_recent_viva_attempts(limit=10)
    viva_trend = progress_store.get_viva_trend()

    if not viva_summary_analytics["questions_answered"]:
        st.info("No viva sessions recorded yet. Complete a viva in the AI Viva tab to see analytics here.")
    else:
        total_sessions = len({row["answered_at"][:10] for row in viva_trend}) if viva_trend else 0

        va_col1, va_col2, va_col3 = st.columns(3)
        with va_col1:
            metric_card(
                "Total Viva Sessions",
                total_sessions,
                "Distinct days with at least one graded viva answer",
                "🎤",
                "teal",
            )
        with va_col2:
            metric_card(
                "Questions Answered",
                viva_summary_analytics["questions_answered"],
                "Total graded viva answers, all-time",
                "❓",
                "blue",
            )
        with va_col3:
            metric_card(
                "Average Score",
                f"{viva_summary_analytics['average_score']}/10",
                f"≈ {round((viva_summary_analytics['average_score'] / 10.0) * 100, 1)}% overall",
                "📊",
                "amber",
            )

        va_col4, va_col5 = st.columns(2)
        with va_col4:
            metric_card(
                "Highest Score",
                f"{viva_summary_analytics['highest_score']}/10",
                "Best single-answer score",
                "🏆",
                "teal",
            )
        with va_col5:
            metric_card(
                "Lowest Score",
                f"{viva_summary_analytics['lowest_score']}/10",
                "Toughest single-answer score",
                "📉",
                "rose",
            )

        render_section_title("🏷️", "Strongest & Weakest Topic")
        st.caption(
            "Viva questions aren't tagged with a topic, so this is based on average score "
            "per difficulty level attempted so far."
        )

        if difficulty_breakdown:
            strongest_topic = max(difficulty_breakdown, key=lambda d: d["average_score"])
            weakest_topic = min(difficulty_breakdown, key=lambda d: d["average_score"])

            st_col1, st_col2 = st.columns(2)
            with st_col1:
                metric_card(
                    "Strongest Topic",
                    strongest_topic["difficulty"],
                    f"Average score {strongest_topic['average_score']}/10 across "
                    f"{strongest_topic['attempts']} question(s)",
                    "💪",
                    "teal",
                )
            with st_col2:
                metric_card(
                    "Weakest Topic",
                    weakest_topic["difficulty"],
                    f"Average score {weakest_topic['average_score']}/10 across "
                    f"{weakest_topic['attempts']} question(s)",
                    "⚠️",
                    "rose",
                )

            st.dataframe(pd.DataFrame(difficulty_breakdown), use_container_width=True, hide_index=True)

        render_section_title("📉", "Score & Confidence Trend")
        if viva_trend:
            trend_df = pd.DataFrame(viva_trend)
            trend_df["answered_at"] = pd.to_datetime(trend_df["answered_at"])
            trend_df = trend_df.set_index("answered_at")

            trend_col1, trend_col2 = st.columns(2)
            with trend_col1:
                st.markdown("**Score Trend**")
                st.line_chart(trend_df["score"])
            with trend_col2:
                st.markdown("**Confidence Trend**")
                st.line_chart(trend_df["confidence"])
        else:
            st.caption("No trend data yet.")

        render_section_title("🕒", "Recent Viva Table")
        if recent_viva_attempts:
            recent_df = pd.DataFrame(recent_viva_attempts).rename(
                columns={
                    "difficulty": "Difficulty",
                    "question": "Question",
                    "score": "Score",
                    "confidence": "Confidence %",
                    "answered_at": "Answered At",
                }
            )
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No recent viva attempts yet.")

        render_section_title("🤖", "AI Performance Insights")
        if st.button("Generate AI Insights", key="viva_analytics_insights_btn"):
            with st.spinner("Analyzing your viva performance..."):
                try:
                    st.session_state["viva_analytics_insights"] = generate_viva_insights(
                        viva_summary_analytics,
                        difficulty_breakdown,
                        recent_viva_attempts,
                    )
                except Exception as e:
                    st.session_state["viva_analytics_insights"] = f"Failed to generate insights.\n\n{e}"

        if st.session_state.get("viva_analytics_insights"):
            st.info(st.session_state["viva_analytics_insights"])
        else:
            st.caption("Click 'Generate AI Insights' for a personalized performance summary.")


with tab_study_recs:
    render_section_title("🧭", "AI Study Recommendations")
    st.caption(
        "Personalized next-step suggestions, blended from your Quiz Generator/Review Due "
        "performance and your AI Viva history."
    )

    dashboard_summary_recs = progress_store.get_dashboard_summary()
    viva_summary_recs = progress_store.get_viva_summary()

    if not dashboard_summary_recs["total_attempts"] and not viva_summary_recs["questions_answered"]:
        st.info(
            "Complete a quiz (Quiz Generator tab) or a viva session (AI Viva tab) to unlock "
            "personalized recommendations."
        )
    else:
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        with rec_col1:
            metric_card(
                "Quiz Accuracy",
                f"{dashboard_summary_recs['overall_accuracy']}%" if dashboard_summary_recs["total_attempts"] else "—",
                f"{dashboard_summary_recs['total_attempts']} quiz/review attempts logged",
                "📝",
                "blue",
            )
        with rec_col2:
            metric_card(
                "Viva Average",
                f"{viva_summary_recs['average_score']}/10" if viva_summary_recs["questions_answered"] else "—",
                f"{viva_summary_recs['questions_answered']} viva question(s) answered",
                "🎤",
                "amber",
            )
        with rec_col3:
            metric_card(
                "Weak Quiz Topics",
                len(dashboard_summary_recs["weak_topics"]),
                "Quiz topics below 60% accuracy",
                "⚠️",
                "rose",
            )

        if st.button("Generate Study Recommendations", key="study_recs_btn", type="primary"):
            with st.spinner("Reviewing your quiz and viva history..."):
                try:
                    st.session_state["study_recommendations"] = generate_study_recommendations(
                        dashboard_summary_recs,
                        viva_summary_recs,
                    )
                except Exception as e:
                    st.session_state["study_recommendations"] = f"Failed to generate recommendations.\n\n{e}"

        if st.session_state.get("study_recommendations"):
            with st.container(border=True):
                st.markdown(st.session_state["study_recommendations"])
        else:
            st.caption("Click 'Generate Study Recommendations' for a personalized study plan.")


with tab_guide:
    render_section_title("🚀", "Project-Ready Guide")
    guide_analytics = get_document_analytics()
    project_report = build_project_report(guide_analytics, provider, model_name)

    guide_cols = st.columns(3)
    guide_items = [
        (
            "1. Upload",
            "Add one or more documents in the sidebar — PDF, Word, PowerPoint, TXT, Markdown, RTF, "
            "or images/scans. Scanned pages and images are read automatically with OCR.",
        ),
        (
            "2. Train",
            "Click Ingest & Train. The app extracts pages, chunks text, embeds it, and stores it in ChromaDB.",
        ),
        (
            "3. Present",
            "Use Chat, Dashboard, Document Tools, Quiz, Review Due, and Mastery tabs to demonstrate "
            "a complete adaptive learning workflow — not just a chatbot.",
        ),
    ]
    for col, (title, body) in zip(guide_cols, guide_items):
        with col:
            render_html(
                textwrap.dedent(
                    f"""
                <div class="guide-card">
                    <div class="topic-title">{escape(title)}</div>
                    <p>{escape(body)}</p>
                </div>
                """
                ).strip()
            )

    render_section_title("📦", "Project Submission Pack")
    report_cols = st.columns([1, 1, 3])
    with report_cols[0]:
        st.download_button(
            "Download Report",
            project_report,
            file_name="tutoai-project-report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with report_cols[1]:
        st.download_button(
            "Download PDF",
            make_pdf_bytes("TutoAI Project Report", project_report),
            file_name="tutoai-project-report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with report_cols[2]:
        st.info("Use this report during viva or project submission to explain the architecture, features, and demo flow.")

    st.markdown("### Suggested Demo Questions")
    st.markdown(
        """
        - Give a summary of the uploaded document.
        - Generate key points from chapter 1.
        - Explain this topic like I am a beginner.
        - What is the difference between these files?
        - Create 10 MCQs from the most important concepts, answer them, then show the Review Due and Mastery tabs.
        """
    )