"""
rag_engine.py
-------------
Backend for TutoAI: document ingestion (PDF, DOCX, PPTX, TXT, MD, RTF,
images, and scanned/OCR documents), ChromaDB vector search, LLM-backed
question answering, document summarization/comparison, and quiz
generation with spaced-repetition logging.

Public API is unchanged so this is a drop-in replacement for app1.py,
which imports these names directly:

    DATA_DIR, PERSIST_DIR, build_index, clear_data, compare_documents,
    generate_document_summary, generate_quiz, get_assistant_modes,
    get_citation_image, get_db_collection, get_document_analytics,
    has_indexed_data, list_uploaded_pdfs, query_knowledge_base,
    save_uploaded_file

Everything else (delete_document, delete_all_documents,
list_uploaded_documents, get_dashboard_summary hooks, etc.) is additive.

NOTE ON UI WIRING: app1.py's file uploader is currently restricted to
`type=["pdf"]` and the sidebar has no per-document Delete button. The
functions below fully support other formats and deletion already —
they just aren't reachable from the current UI until app1.py's
uploader/sidebar are extended to call them.
"""

from __future__ import annotations

import datetime
import gc
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import progress_store

# --- Optional dependencies -------------------------------------------------
# Each of these unlocks one file format / OCR path. Their absence must
# never crash the app -- we just skip that capability and log why.
try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

try:
    import fitz  # PyMuPDF - PDF rasterization for OCR + visual citations
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pytesseract
    from PIL import Image
    import shutil
    import os

    tesseract_path = shutil.which("tesseract")

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

except ImportError:
    pytesseract = None
    Image = None

try:
    import docx  # python-docx
except ImportError:  # pragma: no cover
    docx = None

try:
    from pptx import Presentation  # python-pptx
except ImportError:  # pragma: no cover
    Presentation = None

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:  # pragma: no cover
    rtf_to_text = None

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PERSIST_DIR = "./storage"
DATA_DIR = "./KB"
MODELS_DIR = "./models"
CHROMA_PATH = os.path.join(PERSIST_DIR, "chroma_db")
COLLECTION_NAME = "pdf_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR = os.path.join(
    MODELS_DIR, "models--sentence-transformers--all-MiniLM-L6-v2"
)

ASSISTANT_MODE_INSTRUCTIONS = {
    "Answer briefly": (
        "Answer in 2-4 concise sentences. Prioritize the direct answer and avoid extra detail."
    ),
    "Answer in detail": (
        "Give a thorough explanation with clear structure, relevant details, and supporting context."
    ),
    "Explain like I am a beginner": (
        "Use simple language, define important terms, and build the explanation step by step."
    ),
    "Give a summary": (
        "Summarize the most important information in a compact paragraph or short sections."
    ),
    "Generate key points": (
        "Return the answer as crisp bullet points focused on the most important takeaways."
    ),
}

STOP_WORDS = {
    "about", "above", "after", "again", "against", "being", "below", "between",
    "cannot", "could", "during", "first", "from", "further", "have", "having",
    "into", "more", "most", "other", "over", "same", "should", "some", "such",
    "than", "that", "their", "there", "these", "this", "those", "through",
    "under", "until", "very", "what", "when", "where", "which", "while",
    "with", "within", "would", "your", "document", "documents", "page",
    "chapter", "figure", "table", "section", "using", "used", "also",
}

# Chunking: smaller chunks with more overlap than the original module,
# tuned for better recall on semantically-adjacent topics (fixes the
# "Machine Learning works, Artificial Intelligence / Deep Learning /
# Neural Networks fail" retrieval issue -- smaller, paragraph-aware
# chunks mean fewer unrelated concepts get crammed into one embedding).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Retrieval
DEFAULT_N_RESULTS = 6
DETAILED_N_RESULTS = 8

# LLM call resilience
MAX_LLM_RETRIES = 3
LLM_RETRY_BASE_DELAY_SECONDS = 1.5
OLLAMA_TIMEOUT_SECONDS = 600
GEMINI_TIMEOUT_SECONDS = 60

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".md", ".rtf",
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Progress/spaced-repetition DB (separate from the vector DB)
progress_store.init_db()

# Global singletons (lazy-loaded, cached across calls)
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None


# ---------------------------------------------------------------------------
# Small data holder for extracted, pre-chunk units of text
# ---------------------------------------------------------------------------
@dataclass
class ExtractedUnit:
    """One page / slide / section of raw extracted text, before chunking."""
    text: str
    page: Optional[int] = None
    slide: Optional[int] = None
    section: Optional[int] = None
    extra_meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model / DB initialization (cached singletons)
# ---------------------------------------------------------------------------
def get_embedding_model() -> SentenceTransformer:
    """Returns a process-wide cached embedding model instance so it is
    loaded from disk only once per run, not once per request."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
        kwargs: Dict[str, Any] = {"cache_folder": MODELS_DIR}
        if os.path.exists(os.path.join(EMBEDDING_CACHE_DIR, "refs", "main")):
            kwargs["local_files_only"] = True
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, **kwargs)
    return _embedding_model


def get_db_collection():
    """Returns a process-wide cached ChromaDB collection handle."""
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def has_indexed_data() -> bool:
    """Returns True if the knowledge base currently contains any indexed chunks."""
    try:
        return get_db_collection().count() > 0
    except Exception:
        logger.exception("has_indexed_data: could not read collection count")
        return False


def get_assistant_modes() -> Tuple[str, ...]:
    """Returns the supported answer workflow modes in display order."""
    return tuple(ASSISTANT_MODE_INSTRUCTIONS.keys())


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------
def list_uploaded_pdfs() -> List[str]:
    """Returns uploaded PDF filenames in display order. Kept name-for-name
    for app1.py compatibility -- PDFs only. Use `list_uploaded_documents()`
    for every supported format."""
    return sorted(
        os.path.basename(path) for path in glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    )


def list_uploaded_documents() -> List[str]:
    """Returns every uploaded, supported-format filename in display order."""
    names = []
    for ext in SUPPORTED_EXTENSIONS:
        names.extend(
            os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, f"*{ext}"))
        )
    return sorted(set(names))


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    """Saves an uploaded file (any supported format) to the KB directory
    with a timestamped, sanitized filename. Signature unchanged from the
    original PDF-only version -- it now just works for any extension
    Streamlit hands it."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(uploaded_file.name))
    filename = f"{timestamp}_{safe_name}"
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception:
        logger.exception("save_uploaded_file: failed to write %s", filename)
        raise
    return file_path


def delete_document(filename: str) -> Tuple[bool, str]:
    """Deletes one uploaded document's file and every one of its indexed
    chunks from ChromaDB, without touching any other document. Returns
    (success, message). Never raises.

    Wire this to a per-file "Delete" button once app1.py's sidebar is
    extended -- e.g. `rag_engine.delete_document(pdf_name)`.
    """
    file_path = os.path.join(DATA_DIR, filename)
    try:
        collection = get_db_collection()
        existing = collection.get(where={"source": filename}, include=[])
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

        if os.path.exists(file_path):
            os.remove(file_path)

        logger.info("Deleted document %s (%d chunks removed)", filename, len(ids_to_delete))
        return True, f"Deleted '{filename}' and {len(ids_to_delete)} indexed chunk(s)."
    except Exception as exc:
        logger.exception("delete_document: failed to delete %s", filename)
        return False, f"Could not delete '{filename}': {exc}"


def delete_all_documents() -> Tuple[bool, str]:
    """Deletes every uploaded document, every indexed vector, and resets
    the progress store. This is an alias for `clear_data()` under the
    multi-document naming -- same safe full-reset behavior."""
    try:
        clear_data()
        return True, "All documents and indexed data were removed."
    except Exception as exc:
        logger.exception("delete_all_documents: reset failed")
        return False, f"Reset failed: {exc}"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def _normalize_extracted_text(text: Optional[str]) -> str:
    """Cleans extracted text: rejoins hyphenated line-wraps, collapses
    duplicate whitespace within a line, but preserves paragraph breaks
    (blank lines) so paragraph-aware chunking can still find them."""
    if not text:
        return ""
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)  # de-hyphenate wrapped words
    # Collapse runs of spaces/tabs, but keep paragraph breaks (\n\n) intact.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def _flatten_for_display(text: str) -> str:
    """Single-line version of cleaned text, for snippets/analytics."""
    return re.sub(r"\s+", " ", text or "").strip()


# ---------------------------------------------------------------------------
# Chunking -- paragraph/sentence aware, avoids splitting concepts mid-way
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _split_long_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """Splits a single paragraph that's larger than chunk_size, preferring
    sentence boundaries so a concept isn't cut mid-sentence."""
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size or not current:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
        # A single sentence longer than chunk_size still needs a hard split.
        while len(current) > chunk_size * 1.5:
            pieces.append(current[:chunk_size])
            current = current[chunk_size:]
    if current:
        pieces.append(current)
    return pieces or [paragraph]


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits text into overlapping chunks, greedily packing whole
    paragraphs together up to chunk_size so related sentences stay in
    the same embedding, and only splitting a paragraph internally when
    it alone exceeds chunk_size."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    raw_chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = (
            _split_long_paragraph(paragraph, chunk_size)
            if len(paragraph) > chunk_size
            else [paragraph]
        )
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= chunk_size or not current:
                current = candidate
            else:
                raw_chunks.append(current)
                current = piece
    if current:
        raw_chunks.append(current)

    # Apply overlap between consecutive chunks for retrieval continuity.
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-chunk_overlap:]
        overlapped.append(f"{prev_tail}\n\n{raw_chunks[i]}".strip())
    return overlapped


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
def _ocr_available() -> bool:
    return pytesseract is not None and Image is not None


def _clean_ocr_text(text: str) -> str:
    """OCR output tends to have broken line wraps and stray characters;
    clean it before it enters the normal pipeline."""
    text = re.sub(r"[^\S\n]+", " ", text or "")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[|_~`^]+", "", text)
    return text.strip()


def _ocr_image_bytes(image_bytes: bytes) -> str:
    if not _ocr_available():
        logger.warning("OCR requested but pytesseract/Pillow is not installed; skipping.")
        return ""
    try:
        import io
        image = Image.open(io.BytesIO(image_bytes))
        raw = pytesseract.image_to_string(image)
        return _clean_ocr_text(raw)
    except Exception:
        logger.exception("OCR failed on an image")
        return ""


def _ocr_pdf_page(file_path: str, page_index: int) -> str:
    """Rasterizes one PDF page and OCRs it. Used only when the page has
    no selectable text (i.e. it's a scan)."""
    if not (fitz and _ocr_available()):
        return ""
    try:
        doc = fitz.open(file_path)
        try:
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            return _ocr_image_bytes(pix.tobytes("png"))
        finally:
            doc.close()
    except Exception:
        logger.exception("OCR fallback failed for %s page %s", file_path, page_index)
        return ""


# ---------------------------------------------------------------------------
# Per-format extraction -- each returns a list of ExtractedUnit
# ---------------------------------------------------------------------------
def _extract_pdf(file_path: str) -> List[ExtractedUnit]:
    """Extracts text page by page. Automatically falls back to OCR for
    any page with no selectable text (i.e. scanned pages)."""
    units: List[ExtractedUnit] = []
    reader = PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = _normalize_extracted_text(page.extract_text())
        used_ocr = False
        if not text:
            ocr_text = _ocr_pdf_page(file_path, i)
            if ocr_text:
                text = _normalize_extracted_text(ocr_text)
                used_ocr = True
        if text:
            units.append(ExtractedUnit(text=text, page=i + 1, extra_meta={"ocr": used_ocr}))
    return units


def _extract_docx(file_path: str) -> List[ExtractedUnit]:
    if docx is None:
        raise RuntimeError("python-docx is not installed; cannot read .docx files.")
    document = docx.Document(file_path)
    units: List[ExtractedUnit] = []
    buffer: List[str] = []
    section_num = 1
    for para in document.paragraphs:
        stripped = para.text.strip()
        if stripped:
            buffer.append(stripped)
        # Treat headings (or every ~10 paragraphs) as a section boundary
        # so metadata stays meaningfully granular for long documents.
        is_heading = para.style is not None and "Heading" in (para.style.name or "")
        if is_heading and buffer[:-1]:
            units.append(
                ExtractedUnit(text=_normalize_extracted_text("\n\n".join(buffer[:-1])), section=section_num)
            )
            section_num += 1
            buffer = buffer[-1:]
    if buffer:
        units.append(ExtractedUnit(text=_normalize_extracted_text("\n\n".join(buffer)), section=section_num))

    # Tables often hold important structured facts -- include them too.
    for t_idx, table in enumerate(document.tables, start=1):
        rows_text = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                rows_text.append(" | ".join(cells))
        if rows_text:
            units.append(
                ExtractedUnit(
                    text=_normalize_extracted_text("\n".join(rows_text)),
                    section=section_num,
                    extra_meta={"table": t_idx},
                )
            )
            section_num += 1
    return units


def _extract_pptx(file_path: str) -> List[ExtractedUnit]:
    if Presentation is None:
        raise RuntimeError("python-pptx is not installed; cannot read .pptx files.")
    presentation = Presentation(file_path)
    units: List[ExtractedUnit] = []
    for slide_num, slide in enumerate(presentation.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                texts.append(shape.text_frame.text.strip())
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        texts.append(" | ".join(cells))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            texts.append(f"Speaker notes: {slide.notes_slide.notes_text_frame.text.strip()}")
        if texts:
            units.append(ExtractedUnit(text=_normalize_extracted_text("\n\n".join(texts)), slide=slide_num))
    return units


def _extract_plain_text(file_path: str) -> List[ExtractedUnit]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    text = _normalize_extracted_text(raw)
    return [ExtractedUnit(text=text)] if text else []


def _extract_rtf(file_path: str) -> List[ExtractedUnit]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    if rtf_to_text is not None:
        try:
            text = rtf_to_text(raw)
        except Exception:
            logger.exception("striprtf failed, falling back to a naive strip for %s", file_path)
            text = re.sub(r"\\[a-zA-Z]+\d* ?|[{}]", "", raw)
    else:
        # Naive fallback: strip RTF control words/groups. Good enough for
        # basic documents when striprtf isn't installed.
        text = re.sub(r"\\[a-zA-Z]+\d* ?|[{}]", "", raw)
    text = _normalize_extracted_text(text)
    return [ExtractedUnit(text=text)] if text else []


def _extract_image(file_path: str) -> List[ExtractedUnit]:
    if not _ocr_available():
        raise RuntimeError(
            "OCR is not available (pytesseract/Pillow not installed); cannot read image files."
        )
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    text = _normalize_extracted_text(_ocr_image_bytes(raw_bytes))
    return [ExtractedUnit(text=text, extra_meta={"ocr": True})] if text else []


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".pptx": _extract_pptx,
    ".txt": _extract_plain_text,
    ".md": _extract_plain_text,
    ".rtf": _extract_rtf,
    ".jpg": _extract_image,
    ".jpeg": _extract_image,
    ".png": _extract_image,
    ".bmp": _extract_image,
    ".tif": _extract_image,
    ".tiff": _extract_image,
}


def detect_document_type(file_path: str) -> str:
    """Returns the lowercase extension (including the dot), used for both
    format dispatch and metadata."""
    return os.path.splitext(file_path)[1].lower()


# ---------------------------------------------------------------------------
# Unified ingestion pipeline: extract -> clean -> (OCR) -> chunk
# ---------------------------------------------------------------------------
def process_document(file_path: str) -> List[Dict[str, Any]]:
    """Runs the unified ingestion pipeline for any supported file type and
    returns a flat list of chunk dicts ready for embedding:
        {"id", "text", "metadata": {source, file_type, page, slide,
         section, chunk_number, ...}}

    Never raises for a single bad file -- callers should still wrap this
    per-file so one corrupt document can't stop the whole batch.
    """
    file_name = os.path.basename(file_path)
    file_type = detect_document_type(file_path)
    extractor = _EXTRACTORS.get(file_type)
    if extractor is None:
        raise ValueError(f"Unsupported file type: {file_type}")

    units = extractor(file_path)
    chunks: List[Dict[str, Any]] = []
    seen_hashes: set = set()

    for unit in units:
        if not unit.text:
            continue
        sub_chunks = _split_text(unit.text)
        for j, sub_text in enumerate(sub_chunks):
            # Skip exact-duplicate chunk text within this document (guards
            # against duplicate content if a page/slide repeats verbatim).
            text_hash = hashlib.sha1(sub_text.encode("utf-8", "ignore")).hexdigest()
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            locator = unit.page or unit.slide or unit.section or 0
            chunk_id = f"{file_name}::u{locator}::c{j}::{text_hash[:10]}"
            metadata = {
                "source": file_name,
                "file_type": file_type.lstrip("."),
                "page": unit.page if unit.page is not None else "N/A",
                "slide": unit.slide if unit.slide is not None else "N/A",
                "section": unit.section if unit.section is not None else "N/A",
                "chunk_number": j,
            }
            metadata.update({k: v for k, v in unit.extra_meta.items() if isinstance(v, (str, int, float, bool))})
            chunks.append({"id": chunk_id, "text": sub_text, "metadata": metadata})

    return chunks


def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Kept for backward compatibility with any external caller that
    still imports `process_pdf` directly; delegates to the unified
    pipeline."""
    return process_document(file_path)


# ---------------------------------------------------------------------------
# Topic extraction / confidence scoring (used by analytics + retrieval UI)
# ---------------------------------------------------------------------------
def _extract_topics_from_text(text: str, top_n: int = 6) -> List[Tuple[str, int]]:
    words = re.findall(r"[A-Za-z][A-Za-z-]{3,}", text.lower())
    cleaned = [
        word.strip("-")
        for word in words
        if word not in STOP_WORDS and len(word.strip("-")) > 3
    ]
    return Counter(cleaned).most_common(top_n)


def _confidence_from_distance(distance: Optional[float]) -> int:
    if distance is None:
        return 0
    try:
        score = int(round(100 / (1 + max(float(distance), 0))))
    except (TypeError, ValueError):
        return 0
    return max(1, min(score, 100))


def _confidence_label(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    if score > 0:
        return "Low"
    return "Unknown"


def _trim_text(text: str, limit: int = 360) -> str:
    text = _flatten_for_display(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rsplit(' ', 1)[0]}..."


def _get_index_snapshot() -> Dict[str, List[Any]]:
    """Returns all indexed documents with metadata for analytics and tools."""
    try:
        collection = get_db_collection()
        if collection.count() == 0:
            return {"documents": [], "metadatas": [], "ids": []}
        return collection.get(include=["documents", "metadatas"])
    except Exception:
        logger.exception("_get_index_snapshot: failed to read collection")
        return {"documents": [], "metadatas": [], "ids": []}


def _collect_document_text(document_name: str, limit_chars: int = 9000) -> str:
    snapshot = _get_index_snapshot()
    parts = []
    for doc, meta in zip(snapshot.get("documents", []), snapshot.get("metadatas", [])):
        if meta.get("source") == document_name:
            parts.append(doc)
        if sum(len(part) for part in parts) >= limit_chars:
            break
    return "\n\n".join(parts)[:limit_chars]


def _count_units_in_file(file_path: str, file_type: str) -> int:
    """Approximates a 'pages' count for analytics, generalized across
    formats (pages for PDF, slides for PPTX, sections for DOCX, 1 for
    single-unit formats)."""
    try:
        if file_type == ".pdf":
            return len(PdfReader(file_path).pages)
        if file_type == ".pptx" and Presentation is not None:
            return len(Presentation(file_path).slides)
        if file_type == ".docx" and docx is not None:
            return sum(1 for p in docx.Document(file_path).paragraphs if p.text.strip()) or 1
        return 1
    except Exception:
        logger.exception("_count_units_in_file: failed for %s", file_path)
        return 0


def get_document_analytics() -> Dict[str, Any]:
    """Builds document-level analytics from every uploaded, supported
    document (not just PDFs) and its indexed chunks."""
    doc_names = list_uploaded_documents()
    snapshot = _get_index_snapshot()
    chunk_counts: Counter = Counter()
    page_sets: Dict[str, set] = defaultdict(set)
    doc_text: Dict[str, List[str]] = defaultdict(list)

    for doc, meta in zip(snapshot.get("documents", []), snapshot.get("metadatas", [])):
        source = meta.get("source", "Unknown")
        chunk_counts[source] += 1
        locator = meta.get("page")
        if locator in (None, "N/A"):
            locator = meta.get("slide") if meta.get("slide") not in (None, "N/A") else meta.get("section")
        if locator not in (None, "N/A"):
            page_sets[source].add(locator)
        doc_text[source].append(doc)

    documents = []
    global_text_parts = []
    for name in doc_names:
        path = os.path.join(DATA_DIR, name)
        file_type = detect_document_type(name)
        unit_count = _count_units_in_file(path, file_type)

        text = " ".join(doc_text.get(name, []))
        global_text_parts.append(text)
        topics = _extract_topics_from_text(text, top_n=5)
        documents.append(
            {
                "name": name,
                "pages": unit_count,
                "indexed_pages": len(page_sets.get(name, set())),
                "chunks": chunk_counts.get(name, 0),
                "file_type": file_type.lstrip("."),
                "topics": [{"topic": topic, "count": count} for topic, count in topics],
            }
        )

    top_topics = [
        {"topic": topic, "count": count}
        for topic, count in _extract_topics_from_text(" ".join(global_text_parts), top_n=10)
    ]

    documents_by_topic = []
    for topic_info in top_topics[:8]:
        topic = topic_info["topic"]
        related_docs = []
        for item in documents:
            topic_count = next(
                (entry["count"] for entry in item["topics"] if entry["topic"] == topic), 0
            )
            if topic_count:
                related_docs.append({"document": item["name"], "mentions": topic_count})
        documents_by_topic.append({"topic": topic, "documents": related_docs})

    return {
        "documents": documents,
        "total_pages": sum(item["pages"] for item in documents),
        "total_chunks": sum(item["chunks"] for item in documents),
        "top_topics": top_topics,
        "documents_by_topic": documents_by_topic,
    }


def generate_document_summary(document_name: Optional[str] = None) -> str:
    """Generates a concise summary for one document or the whole knowledge base."""
    try:
        collection = get_db_collection()
        if collection.count() == 0:
            return "No documents are indexed yet. Upload documents and click Ingest & Train first."
    except Exception as exc:
        logger.exception("generate_document_summary: collection unavailable")
        return f"Could not reach the knowledge base: {exc}"

    if document_name:
        context_text = _collect_document_text(document_name)
        title = document_name
    else:
        snapshot = _get_index_snapshot()
        context_text = "\n\n".join(snapshot.get("documents", [])[:12])[:9000]
        title = "the uploaded documents"

    if not context_text.strip():
        return "No readable indexed text was found for that document."

    prompt = f"""
    Summarize {title} for a student. Use only the context below.
    Include:
    - Main idea
    - Important concepts
    - What to revise before an exam

    Context:
    {context_text}
    """

    try:
        return _generate_with_active_provider(prompt)
    except Exception as e:
        logger.exception("generate_document_summary: generation failed")
        return f"Error generating summary: {e}"


def compare_documents(document_a: str, document_b: str) -> str:
    """Compares two indexed documents (any supported format) and returns
    well-formatted Markdown covering common topics, differences, unique
    concepts, a summary, and key findings."""
    if not document_a or not document_b or document_a == document_b:
        return "Choose two different indexed documents to compare."

    text_a = _collect_document_text(document_a, limit_chars=6000)
    text_b = _collect_document_text(document_b, limit_chars=6000)

    if not text_a.strip() or not text_b.strip():
        return "Both documents need indexed text before they can be compared."

    prompt = f"""
    Compare these two documents for a student. Use only the provided excerpts.
    Respond in Markdown with these exact sections:

    ## Summary
    A short paragraph on what each document is about overall.

    ## Common Topics
    Bullet points of concepts covered by both documents.

    ## Differences
    Bullet points contrasting how the documents diverge in approach, depth, or conclusions.

    ## Unique Concepts
    - **{document_a}**: concepts found only here
    - **{document_b}**: concepts found only here

    ## Key Findings
    2-4 bullet points a student should remember before an exam.

    Document A: {document_a}
    {text_a}

    Document B: {document_b}
    {text_b}
    """

    try:
        return _generate_with_active_provider(prompt)
    except Exception as e:
        logger.exception("compare_documents: generation failed")
        return f"Error comparing documents: {e}"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def build_index() -> str:
    """Reads every supported document in DATA_DIR, creates embeddings, and
    rebuilds the vector DB from scratch (a full rebuild is what avoids
    duplicate entries across re-ingestions -- simpler and more reliable
    than trying to diff against the previous state)."""
    global _chroma_client, _collection

    get_db_collection()  # ensures _chroma_client is initialized

    try:
        _chroma_client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        logger.info("No existing collection to delete (safe to ignore): %s", e)

    _collection = _chroma_client.get_or_create_collection(COLLECTION_NAME)
    collection = _collection
    model = get_embedding_model()

    all_files = [
        p for p in glob.glob(os.path.join(DATA_DIR, "*"))
        if detect_document_type(p) in SUPPORTED_EXTENSIONS
    ]
    all_chunks: List[Dict[str, Any]] = []
    failed_files: List[str] = []

    for file_path in all_files:
        try:
            chunks = process_document(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.exception("build_index: failed to process %s", file_path)
            failed_files.append(f"{os.path.basename(file_path)} ({e})")

    if not all_chunks:
        if failed_files:
            return "No documents could be indexed. Failed: " + "; ".join(failed_files)
        return "No documents found."

    texts = [chunk["text"] for chunk in all_chunks]
    ids = [chunk["id"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]

    logger.info("Generating embeddings for %d chunks...", len(texts))
    try:
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
    except Exception as e:
        logger.exception("build_index: embedding generation failed")
        return f"Embedding generation failed: {e}"

    try:
        # Chroma can reject very large single .add() calls; batch defensively.
        batch_size = 500
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            collection.add(
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
    except Exception as e:
        logger.exception("build_index: ChromaDB write failed")
        return f"Failed to store embeddings in ChromaDB: {e}"

    summary = f"Index built successfully: {len(texts)} chunks from {len(all_files) - len(failed_files)} document(s)."
    if failed_files:
        summary += " Skipped (failed): " + "; ".join(failed_files)
    return summary


def clear_data() -> None:
    """Clears physical files and DB, and forces ChromaDB to release its
    internal file handles first (required on Windows, where open files
    cannot be deleted while still locked by another process/handle)."""
    global _chroma_client, _collection

    _chroma_client = None
    _collection = None

    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception as e:
        logger.info("Could not clear Chroma system cache (safe to ignore): %s", e)

    gc.collect()

    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
        except PermissionError as e:
            logger.warning("Could not fully delete %s: %s", PERSIST_DIR, e)
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        except Exception:
            logger.exception("Unexpected error clearing %s", PERSIST_DIR)

    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                os.unlink(file_path)
            except OSError as e:
                logger.warning("Could not delete %s: %s", file_path, e)

    os.makedirs(PERSIST_DIR, exist_ok=True)

    try:
        progress_store.reset_progress()
    except Exception:
        logger.exception("Could not reset progress store (safe to ignore)")


# ---------------------------------------------------------------------------
# LLM providers (Gemini + Ollama) with retry, timeout, never-crash
# ---------------------------------------------------------------------------
def _retry(fn, *, max_attempts: int = MAX_LLM_RETRIES, base_delay: float = LLM_RETRY_BASE_DELAY_SECONDS):
    """Runs `fn()` with exponential-backoff retries. Re-raises the last
    exception if every attempt fails, so callers keep their existing
    try/except behavior."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s -- retrying in %.1fs",
                    attempt, max_attempts, exc, delay,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def query_gemini(prompt: str, api_key: Optional[str], model_name: str = "gemini-2.0-flash") -> str:
    if genai is None:
        raise RuntimeError("The 'google-genai' package is not installed; cannot use Gemini.")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file to use Gemini.")

    def _call():
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        if not getattr(response, "text", None):
            raise RuntimeError("Gemini returned an empty response.")
        return response.text

    return _retry(_call)


def _get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _ollama_model_available(model_name: str) -> bool:
    """Check whether a model is already pulled locally in Ollama."""
    try:
        resp = requests.post(
            f"{_get_ollama_base_url()}/api/show", json={"model": model_name}, timeout=5
        )
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def query_ollama(prompt: str, model_name: str = "llama3.2:3b") -> str:
    if not _ollama_model_available(model_name):
        raise RuntimeError(
            f"Ollama model '{model_name}' is not available. "
            "Make sure Ollama is running (`ollama serve`) and the model is "
            f"pulled (`ollama pull {model_name}`)."
        )

    url = f"{_get_ollama_base_url()}/api/generate"
    data = {"model": model_name, "prompt": prompt, "stream": False}

    def _call():
        try:
            response = requests.post(url, json=data, timeout=OLLAMA_TIMEOUT_SECONDS)
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama timed out after {OLLAMA_TIMEOUT_SECONDS}s.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {_get_ollama_base_url()}. "
                "Make sure `ollama serve` is running."
            )
        if response.status_code != 200:
            raise RuntimeError(f"Error from Ollama: {response.text}")
        try:
            return response.json()["response"]
        except (ValueError, KeyError) as exc:
            raise RuntimeError(f"Ollama returned an unexpected payload: {exc}")

    return _retry(_call)


def _generate_with_active_provider(prompt: str) -> str:
    """Routes a prompt to whichever provider is configured in .env.
    Never crashes the caller for a transient failure -- retries are
    handled inside query_gemini/query_ollama."""
    provider = os.getenv("LLM_PROVIDER", "Ollama").strip().lower()

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return query_gemini(prompt, api_key, model_name)
    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        return query_ollama(prompt, model_name)

    raise RuntimeError("LLM_PROVIDER must be either 'Ollama' or 'Gemini'.")


# ---------------------------------------------------------------------------
# Visual citations -- render the exact PDF page with the cited text highlighted
# ---------------------------------------------------------------------------
def get_citation_image(source_file: str, page_num: Any, snippet: str, zoom: float = 1.6) -> Optional[bytes]:
    """Renders the given page of source_file as a PNG image with the cited
    snippet highlighted (where it can be located). Returns PNG bytes, or
    None if the page/file can't be found, isn't a PDF, or PyMuPDF isn't
    installed. Never raises."""
    if fitz is None:
        return None

    file_path = os.path.join(DATA_DIR, source_file)
    if not os.path.exists(file_path) or detect_document_type(file_path) != ".pdf":
        return None

    try:
        doc = fitz.open(file_path)
    except Exception:
        logger.exception("Could not open %s for citation rendering", file_path)
        return None

    try:
        if not isinstance(page_num, int) or page_num < 1 or page_num > len(doc):
            return None
        page = doc[page_num - 1]

        candidates = [
            (snippet or "").strip()[:120],
            " ".join((snippet or "").split())[:60],
            " ".join((snippet or "").split())[:30],
        ]
        rects = []
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            rects = page.search_for(candidate)
            if rects:
                break

        for r in rects:
            page.add_highlight_annot(r)

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    except Exception:
        logger.exception("Citation rendering failed for %s page %s", source_file, page_num)
        return None
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# RAG query
# ---------------------------------------------------------------------------
def query_knowledge_base(
    query: str,
    response_mode: str = "Answer briefly",
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Answers `query` using retrieved context, tagged with source cards,
    confidence/relevance scores, and follow-up chat continuity. Never
    raises -- retrieval or generation failures are surfaced in
    `response["response"]` as a readable message."""
    try:
        collection = get_db_collection()
        if collection.count() == 0:
            return {
                "response": "No documents are indexed yet. Upload documents and click Ingest & Train first.",
                "source_nodes": [],
                "response_mode": response_mode,
                "context_used": "",
            }
    except Exception as exc:
        logger.exception("query_knowledge_base: knowledge base unavailable")
        return {
            "response": f"The knowledge base is currently unavailable: {exc}",
            "source_nodes": [],
            "response_mode": response_mode,
            "context_used": "",
        }

    try:
        model = get_embedding_model()
        query_embed = model.encode([query]).tolist()
        n_results = (
            DETAILED_N_RESULTS if response_mode in {"Answer in detail", "Give a summary"} else DEFAULT_N_RESULTS
        )
        results = collection.query(
            query_embeddings=query_embed,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.exception("query_knowledge_base: retrieval failed")
        return {
            "response": f"Retrieval from the knowledge base failed: {exc}",
            "source_nodes": [],
            "response_mode": response_mode,
            "context_used": "",
        }

    context_text = ""
    sources: List[Dict[str, Any]] = []

    if results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            clean_doc = _flatten_for_display(doc)
            distance = None
            if results.get("distances") and results["distances"][0]:
                distance = results["distances"][0][i]
            relevance = _confidence_from_distance(distance)
            locator = meta.get("page")
            if locator in (None, "N/A"):
                locator = meta.get("slide", "N/A")
            context_text += f"\n---\nSource: {meta.get('source', 'Unknown')} (Page {locator})\nContent: {clean_doc}\n"
            sources.append(
                {
                    "source": meta.get("source", "Unknown"),
                    "page": locator,
                    "snippet": _trim_text(clean_doc, limit=420),
                    "context": clean_doc,
                    "relevance": relevance,
                    "relevance_label": _confidence_label(relevance),
                    "distance": distance,
                }
            )

    if not sources:
        return {
            "response": "I couldn't find any relevant content in the indexed documents for this question.",
            "source_nodes": [],
            "response_mode": response_mode,
            "context_used": "",
        }

    mode_instruction = ASSISTANT_MODE_INSTRUCTIONS.get(
        response_mode, ASSISTANT_MODE_INSTRUCTIONS["Answer briefly"]
    )

    follow_up_context = ""
    if chat_history:
        recent_turns = []
        for item in chat_history[-6:]:
            role = item.get("role", "user")
            content = _trim_text(item.get("content", ""), limit=500)
            if content:
                recent_turns.append(f"{role}: {content}")
        follow_up_context = "\n".join(recent_turns)

    prompt = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
    If the answer is not in the context, say "I couldn't find the answer in the provided documents."
    If the user asks a follow-up question, use the recent conversation only to understand what the follow-up refers to.

    Response mode:
    {response_mode}

    Response instructions:
    {mode_instruction}

    Context:
    {context_text}

    Recent conversation:
    {follow_up_context}

    Question:
    {query}
    """

    try:
        response_text = _generate_with_active_provider(prompt)
    except Exception as exc:
        logger.exception("query_knowledge_base: generation failed")
        response_text = f"Error generating response: {exc}"

    return {
        "response": response_text,
        "source_nodes": sources,
        "response_mode": response_mode,
        "context_used": context_text,
    }


# ---------------------------------------------------------------------------
# Quiz generation -- robust JSON parsing with repair + retry
# ---------------------------------------------------------------------------
def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.replace("```", "")


def _fix_invalid_json_escapes(text: str) -> str:
    r"""Escapes any backslash that isn't already a valid JSON escape.

    LLMs occasionally emit a lone backslash inside a JSON string (e.g.
    from LaTeX-like notation, a Windows-style path, or a stray "\d"/"\p"),
    which is not legal JSON and makes json.loads() raise
    'Invalid \escape'. Valid JSON escapes (\", \\, \/, \b, \f, \n,
    \r, \t, \uXXXX) are left untouched; anything else gets its
    backslash doubled so it parses as a literal backslash instead of
    crashing the parser.
    """
    return re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", text)


def _extract_json_array(text: str) -> str:
    """Extracts the outermost, bracket-balanced JSON array from arbitrary
    surrounding text (commentary, partial markdown, etc.)."""
    start = text.find("[")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unbalanced (likely truncated) -- return what we have and let the
    # repair step try to close it.
    return text[start:]


def _repair_json_array(text: str) -> str:
    """Fixes the small formatting errors LLMs commonly introduce: trailing
    commas, smart quotes, unterminated final object/array, single
    quotes used as string delimiters."""
    repaired = text.strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)  # trailing commas

    # Balance brackets/braces if the model truncated output.
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    if open_braces > 0:
        repaired += "}" * open_braces
    if open_brackets > 0:
        repaired += "]" * open_brackets
    return repaired


def _parse_quiz_json(raw_text: str) -> List[Dict[str, Any]]:
    """Turns raw LLM output into a valid Python list, tolerating markdown
    fences, extra commentary, and small JSON formatting errors. Never
    raises -- returns [] if nothing usable could be salvaged."""
    text = _strip_markdown_fences((raw_text or "").strip())
    array_text = _extract_json_array(text)
    if not array_text:
        return []

    for candidate in (array_text, _repair_json_array(array_text)):
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue
    return []


def _validate_mcq_question(question: Dict[str, Any]) -> bool:
    """A usable MCQ question needs question text, at least 2 options, and
    a correct_answer that resolves to exactly one of those options."""
    if not question.get("question") or not isinstance(question.get("options"), list):
        return False
    options = question["options"]
    if len(options) < 2:
        return False
    letter = progress_store.normalize_mcq_letter(question.get("correct_answer"), options)
    return bool(letter) and (ord(letter) - ord("A")) < len(options)


def _validate_descriptive_question(question: Dict[str, Any]) -> bool:
    return bool(question.get("question")) and bool(question.get("model_answer"))


def _build_quiz_prompt(topic: str, num_questions: int, q_type: str, context_text: str) -> str:
    if q_type == "MCQ":
        return f"""
Based only on the text below, generate exactly {num_questions} multiple-choice questions about "{topic}".

Return ONLY a valid JSON array. No markdown fences, no commentary, no text before or after the array.

Format:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct_answer": "A",
    "explanation": "..."
  }}
]

Rules:
1. Exactly four options labelled A, B, C and D.
2. Exactly ONE option must be correct.
3. "correct_answer" must be ONLY a single letter: A, B, C, or D.
4. Double-check that the correct_answer letter matches the actually-correct option before returning the JSON.
5. Do not include markdown or any extra text outside the JSON array.

Context:
{context_text}
"""
    return f"""
Based only on the text below, generate exactly {num_questions} descriptive questions with model answers about "{topic}".

Return ONLY a valid JSON array. No markdown fences, no commentary, no text before or after the array.

Format:
[
  {{"question": "...", "model_answer": "..."}}
]

Context:
{context_text}
"""


def _gather_quiz_context(topic: str, collection) -> str:
    """Retrieves relevant chunks for quiz generation. Uses more chunks and
    a higher per-chunk character budget than the original implementation
    (which capped context at 2 chunks x 800 chars and was the main cause
    of unreliable generation for anything beyond the most common topic)."""
    model = get_embedding_model()
    query_embed = model.encode([topic]).tolist()
    results = collection.query(query_embeddings=query_embed, n_results=DETAILED_N_RESULTS)

    context_parts = []
    if results.get("documents") and results["documents"][0]:
        for doc in results["documents"][0]:
            clean = _flatten_for_display(doc)
            if clean:
                context_parts.append(clean[:1200])
    return "\n\n".join(context_parts).strip()


def generate_quiz(
    topic: str, num_questions: int = 5, q_type: str = "MCQ"
) -> Tuple[List[Dict[str, Any]], Optional[str], str]:
    """Generates a quiz as structured data (never raw markdown), retrying
    generation automatically up to MAX_LLM_RETRIES times if parsing or
    validation fails, and persists each question as a spaced-repetition
    card. Never crashes -- always returns a valid Python list (possibly
    empty) plus a human-readable status message.

    Returns: (questions, provider_used, message)
      questions: list of dicts. For MCQ:
          {"question", "options", "correct_answer", "explanation", "card_id"}
        For Descriptive:
          {"question", "model_answer", "card_id"}
      provider_used: the LLM_PROVIDER env value that generated the quiz,
        or None if generation failed / no context was found.
      message: human-readable status, only meaningful when questions is
        empty (e.g. "No documents indexed yet.").
    """
    num_questions = max(1, min(20, int(num_questions or 5)))

    try:
        collection = get_db_collection()
        if collection.count() == 0:
            return [], None, "No documents are indexed yet. Upload documents and click Ingest & Train first."
    except Exception as exc:
        logger.exception("generate_quiz: knowledge base unavailable")
        return [], None, f"The knowledge base is currently unavailable: {exc}"

    try:
        context_text = _gather_quiz_context(topic, collection)
    except Exception as exc:
        logger.exception("generate_quiz: retrieval failed")
        return [], None, f"Could not retrieve context for that topic: {exc}"

    if not context_text:
        return [], None, "No relevant content found in the knowledge base for this topic."

    provider_used = os.getenv("LLM_PROVIDER", "Ollama")
    validator = _validate_mcq_question if q_type == "MCQ" else _validate_descriptive_question

    questions: List[Dict[str, Any]] = []
    last_error: Optional[str] = None

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        prompt = _build_quiz_prompt(topic, num_questions, q_type, context_text)
        try:
            raw = _generate_with_active_provider(prompt)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("generate_quiz: LLM call failed on attempt %d/%d: %s", attempt, MAX_LLM_RETRIES, exc)
            continue

        parsed = _parse_quiz_json(raw)
        valid = [q for q in parsed if isinstance(q, dict) and validator(q)]

        if valid:
            questions = valid
            break

        last_error = "The model didn't return a parseable/valid quiz."
        logger.warning(
            "generate_quiz: attempt %d/%d produced no valid questions (parsed=%d)",
            attempt, MAX_LLM_RETRIES, len(parsed),
        )

    if not questions:
        return [], provider_used, (
            "Could not generate a valid quiz after several attempts "
            f"({last_error or 'unknown error'}). Try again or use a different topic."
        )

    questions = questions[:num_questions]

    for q in questions:
        if q_type == "MCQ":
            card_id = progress_store.add_card(
                topic=topic,
                question_type="MCQ",
                question=q.get("question", ""),
                options=q.get("options", []),
                correct_answer=q.get("correct_answer", ""),
                explanation=q.get("explanation", ""),
            )
            # Normalize in-memory too, so the *first* grading (before the
            # DB round-trip) uses the exact same clean letter that Review
            # Due will later read back -- this is what actually closes the
            # "quiz says correct, review due says incorrect" gap.
            q["correct_answer"] = progress_store.normalize_mcq_letter(
                q.get("correct_answer", ""), q.get("options", [])
            )
        else:
            card_id = progress_store.add_card(
                topic=topic,
                question_type="Descriptive",
                question=q.get("question", ""),
                options=[],
                correct_answer=q.get("model_answer", ""),
                explanation="",
            )
        q["card_id"] = card_id

    return questions, provider_used, ""

def generate_learning_path(topic, level, goal, duration):
    prompt = """
You are an AI tutor.

Based ONLY on the uploaded documents, create a study roadmap.

Topic: {topic}
Current Level: {level}
Goal: {goal}
Duration: {duration}


Rules:

1. Arrange the concepts from easiest to hardest.
2. Each topic must contain:
   - topic
   - time
3. Return ONLY a valid JSON array.
4. Do NOT write explanations.
5. Do NOT use markdown.
6. Do NOT wrap the output inside ```json or ```.

Output format:

[
  {
    "topic": "Introduction",
    "time": "20 min"
  }
]

Example:

[
  {
    "topic": "Introduction",
    "time": "20 min"
  },
  {
    "topic": "Machine Learning",
    "time": "35 min"
  },
  {
    "topic": "Neural Networks",
    "time": "45 min"
  }
]
"""

    response = query_knowledge_base(
        prompt,
        "Answer in detail"
    )

    return response["response"]

def delete_document(filename):
    import os

    file_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)

    build_index()

    return True

def generate_viva_questions(level, num_questions, topic=""):
    """
    Generate viva questions from the uploaded documents.
    Returns a Python list instead of a raw JSON string.

    `topic` is optional (default "") so any existing caller that passes
    only (level, num_questions) keeps working unchanged. When a topic is
    supplied, questions are grounded in retrieval for that topic
    specifically, reusing the same context-gathering helper as the Quiz
    Generator (_gather_quiz_context) instead of the generic whole-prompt
    retrieval used previously.

    Retries generation up to MAX_LLM_RETRIES times (same constant/pattern
    generate_quiz already uses), keeping the largest valid batch seen
    across attempts, so requesting e.g. 5 questions doesn't silently end
    up with only 2 just because one generation came back short/malformed.
    """
    num_questions = max(1, min(20, int(num_questions or 5)))

    context_text = ""
    if topic:
        try:
            collection = get_db_collection()
            if collection.count() > 0:
                context_text = _gather_quiz_context(topic, collection)
        except Exception:
            context_text = ""

    def _build_prompt():
        context_block = (
            f"\nRelevant document context on '{topic}':\n{context_text}\n" if context_text else ""
        )
        topic_line = f" specifically about: {topic}" if topic else ""
        return f"""
You are an experienced university viva examiner.

Using ONLY the uploaded documents, generate {num_questions} viva questions{topic_line}.
{context_block}
Difficulty Level:
{level}

Rules:

1. Questions must come ONLY from the uploaded documents.
2. Do not use outside knowledge.
3. Include conceptual and application-based questions.
4. Avoid repeating questions.
5. Generate EXACTLY {num_questions} question(s) -- no more, no fewer.

Return ONLY a valid JSON array.

Do NOT write explanations.

Do NOT use markdown.

Do NOT wrap inside ```json.

Output format:

[
    {{
        "question":"What is OCR?"
    }},
    {{
        "question":"Explain Retrieval-Augmented Generation."
    }}
]
"""

    def _parse(raw_text):
        raw_text = _strip_markdown_fences(raw_text or "").strip()
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start != -1 and end != -1:
            raw_text = raw_text[start:end + 1]

        try:
            parsed = json.loads(_fix_invalid_json_escapes(raw_text))
        except Exception:
            repair_prompt = f"""
Convert the following into VALID JSON.

Return ONLY JSON.

{raw_text}
"""
            try:
                repaired = _strip_markdown_fences(_generate_with_active_provider(repair_prompt))
                start = repaired.find("[")
                end = repaired.rfind("]")
                if start != -1 and end != -1:
                    repaired = repaired[start:end + 1]
                parsed = json.loads(_fix_invalid_json_escapes(repaired))
            except Exception:
                fallback_questions = re.findall(
                    r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text
                )
                parsed = [
                    {"question": q.replace('\\"', '"').replace("\\\\", "\\")}
                    for q in fallback_questions
                ]

        if not isinstance(parsed, list):
            return []
        return [q for q in parsed if isinstance(q, dict) and str(q.get("question", "")).strip()]

    best_questions: List[Dict[str, Any]] = []
    for _attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            if context_text:
                raw = _generate_with_active_provider(_build_prompt())
            else:
                raw = query_knowledge_base(_build_prompt(), "")["response"]
        except Exception:
            continue

        parsed = _parse(raw)
        if len(parsed) > len(best_questions):
            best_questions = parsed
        if len(best_questions) >= num_questions:
            break

    if not best_questions:
        raise ValueError("Could not generate any viva questions from the uploaded documents. Please try again.")

    return best_questions[:num_questions]

import re


def evaluate_viva_answer(question, answer):
    """
    Evaluate a student's answer using the uploaded documents.
    """

    prompt = f"""
You are a university viva examiner.

Question:
{question}

Student Answer:
{answer}

Evaluate ONLY using the uploaded documents.

Return exactly in this format:

Score: X/10

Correct Answer:
...

Strengths:
...

Missing Points:
...

Suggestions:
...
"""

    response = query_knowledge_base(
        prompt,
        ""
    )

    return response["response"]

def extract_score(feedback):
    """
    Extract numeric score from evaluation text.
    """

    match = re.search(r"Score\s*:\s*(\d+)", feedback)

    if match:
        return int(match.group(1))

    return 0


def analyze_viva_confidence(question, answer, evaluation):
    """
    Score a viva answer along four extra dimensions -- confidence, accuracy,
    completeness, and communication -- each 0-100.

    This is purely additive: it does NOT change evaluate_viva_answer's
    existing "Score: X/10" grading in any way, and is only ever called
    alongside it, never instead of it. Never raises -- on any failure
    (LLM error, bad JSON, etc.) it falls back to a dict of zeros so a
    flaky analysis call can't break the viva flow.
    """

    prompt = f"""
You are a university viva examiner reviewing your own evaluation of a student's answer.

Question:
{question}

Student Answer:
{answer}

Your Evaluation:
{evaluation}

Rate the student's answer on these four dimensions, each as an integer from 0 to 100:

- confidence: how confidently and assertively the answer was delivered (tone/phrasing, not correctness)
- accuracy: how factually correct the answer is against the uploaded documents
- completeness: how much of the expected answer was covered
- communication: how clearly and coherently the answer was expressed

Return ONLY a valid JSON object, nothing else.

Do NOT use markdown.

Do NOT wrap inside ```json.

Output format:

{{"confidence": 0, "accuracy": 0, "completeness": 0, "communication": 0}}
"""

    result = {"confidence": 0, "accuracy": 0, "completeness": 0, "communication": 0}

    try:
        response = query_knowledge_base(prompt, "")
        raw = _strip_markdown_fences(response["response"].strip())

        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end != -1:
            raw = raw[start:end + 1]

        parsed = json.loads(raw)

        def _clamp(value):
            try:
                value = int(round(float(value)))
            except (TypeError, ValueError):
                return 0
            return max(0, min(100, value))

        for key in result:
            result[key] = _clamp(parsed.get(key))

    except Exception:
        # Fail soft -- confidence analysis is a bonus metric, never worth
        # breaking the viva flow over.
        pass

    return result


# =====================================================
# Viva Analytics -- AI Performance Insights
# =====================================================
def generate_viva_insights(viva_summary, difficulty_breakdown, recent_attempts):
    """
    AI-generated performance insight summary for the Viva Analytics tab.

    Purely additive: summarizes numbers already computed by
    progress_store (get_viva_summary / get_viva_difficulty_breakdown /
    get_recent_viva_attempts) and reuses the same LLM routing as
    generate_document_summary. Does not touch viva generation/grading
    logic in any way. Never raises.
    """
    if not viva_summary.get("questions_answered"):
        return "Not enough viva data yet. Complete at least one viva session to see AI insights here."

    breakdown_lines = "\n".join(
        f"- {d['difficulty']}: {d['attempts']} question(s), average score {d['average_score']}/10"
        for d in difficulty_breakdown
    ) or "No breakdown available."

    recent_lines = "\n".join(
        f"- Q: {str(r['question'])[:120]} | Score: {r['score']}/10 | Confidence: {r['confidence']}%"
        for r in recent_attempts[:5]
    ) or "No recent attempts."

    prompt = f"""
You are an academic performance coach reviewing a student's oral viva exam history.

Overall stats:
- Questions answered: {viva_summary.get('questions_answered', 0)}
- Average score: {viva_summary.get('average_score', 0)}/10
- Highest score: {viva_summary.get('highest_score', 0)}/10
- Lowest score: {viva_summary.get('lowest_score', 0)}/10
- Average confidence: {viva_summary.get('average_confidence', 0)}%
- Average accuracy: {viva_summary.get('average_accuracy', 0)}%
- Average completeness: {viva_summary.get('average_completeness', 0)}%
- Average communication: {viva_summary.get('average_communication', 0)}%

Performance by difficulty level:
{breakdown_lines}

Most recent attempts:
{recent_lines}

Write a short (4-6 sentence) performance insight summary for the student. Mention
their strongest and weakest area, one concrete improvement tip, and an
encouraging closing line. Plain prose only, no markdown headers.
"""

    try:
        return _generate_with_active_provider(prompt)
    except Exception as e:
        return f"AI insights are temporarily unavailable ({e})."


# =====================================================
# AI Study Recommendations
# =====================================================
def generate_study_recommendations(dashboard_summary, viva_summary):
    """
    AI-generated "what to study next" recommendations, blending existing
    quiz mastery data (progress_store.get_dashboard_summary) with existing
    viva performance data (progress_store.get_viva_summary).

    Purely additive: reads already-computed summaries and reuses the same
    LLM routing as generate_document_summary / generate_viva_insights.
    Does not modify quiz, viva, or dashboard logic. Never raises.
    """
    if not dashboard_summary.get("total_attempts") and not viva_summary.get("questions_answered"):
        return "Complete a quiz or a viva session first so there's enough data to base recommendations on."

    quiz_lines = "\n".join(
        f"- {t['topic']}: {t['accuracy']}% accuracy over {t['total']} attempt(s)"
        for t in dashboard_summary.get("topic_breakdown", [])
    ) or "No quiz attempts yet."

    prompt = f"""
You are a study coach. Based on the student's quiz and viva performance data below,
recommend what they should study next.

Quiz topic breakdown:
{quiz_lines}

Quiz overall accuracy: {dashboard_summary.get('overall_accuracy', 0)}%
Strong quiz topics: {", ".join(dashboard_summary.get('strong_topics', [])) or "None yet"}
Weak quiz topics: {", ".join(dashboard_summary.get('weak_topics', [])) or "None yet"}

Viva questions answered: {viva_summary.get('questions_answered', 0)}
Viva average score: {viva_summary.get('average_score', 0)}/10
Viva average confidence: {viva_summary.get('average_confidence', 0)}%

Give the student:
1. Their top 2-3 priority topics to review next, with a one-line reason each.
2. One suggested next action (e.g. take a quiz, do a viva, review due cards).

Keep it under 150 words, plain prose, no markdown headers.
"""

    try:
        return _generate_with_active_provider(prompt)
    except Exception as e:
        return f"AI recommendations are temporarily unavailable ({e})."