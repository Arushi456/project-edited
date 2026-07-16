"""
progress_store.py
------------------
SQLite-backed storage for TutoAI's learning-progress layer:

  1. Quiz questions turned into spaced-repetition "cards" (SM-2 algorithm)
  2. A log of every quiz / review attempt, tagged by topic (mastery dashboard)

Public API is unchanged from the original module so it stays a drop-in
replacement for app1.py:

    init_db()
    add_card(topic, question_type, question, options, correct_answer, explanation="")
    record_attempt(card_id, is_correct)
    get_due_cards(limit=20)
    get_due_count()
    get_topic_mastery()
    reset_progress()

Everything else (normalization helpers, extra analytics) is additive and
safe to ignore from the caller's side.

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import datetime
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

PERSIST_DIR = "./storage"
DB_PATH = os.path.join(PERSIST_DIR, "progress.db")

# Cards with this many correct repetitions in a row (and no recent miss)
# are considered "mastered" for dashboard purposes.
MASTERED_REPETITIONS_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------
@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Context-managed connection: commits on success, rolls back on error,
    always closes. Centralizing this avoids leaked connections/handles,
    which is the most common source of 'database is locked' errors."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _get_conn() -> sqlite3.Connection:
    """Legacy-style single connection getter, kept for any external code
    that may still call it directly. Prefer `_connect()` internally."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def init_db() -> None:
    """Creates tables (and migrates missing columns) if they don't exist
    yet. Safe to call multiple times / on every startup."""
    try:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    question_type TEXT NOT NULL,   -- 'MCQ' or 'Descriptive'
                    question TEXT NOT NULL,
                    options TEXT,                  -- JSON list, empty for descriptive
                    correct_answer TEXT,           -- letter for MCQ, model answer for descriptive
                    explanation TEXT,
                    created_at TEXT NOT NULL,
                    ease_factor REAL DEFAULT 2.5,
                    interval INTEGER DEFAULT 0,
                    repetitions INTEGER DEFAULT 0,
                    next_review TEXT NOT NULL,
                    last_result TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    is_correct INTEGER NOT NULL,
                    answered_at TEXT NOT NULL
                )
                """
            )
            # Additive migration: extra column so the original letter AND
            # the full option text are both available for robust grading,
            # without changing the meaning of the existing `correct_answer`
            # column that app1.py already reads.
            if not _column_exists(conn, "cards", "correct_option_text"):
                conn.execute("ALTER TABLE cards ADD COLUMN correct_option_text TEXT")
            if not _column_exists(conn, "cards", "source_document"):
                conn.execute("ALTER TABLE cards ADD COLUMN source_document TEXT")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS viva_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    difficulty TEXT,
                    question TEXT NOT NULL,
                    answer TEXT,
                    evaluation TEXT,
                    score INTEGER DEFAULT 0,
                    confidence INTEGER DEFAULT 0,
                    accuracy INTEGER DEFAULT 0,
                    completeness INTEGER DEFAULT 0,
                    communication INTEGER DEFAULT 0,
                    answered_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viva_answered_at ON viva_attempts(answered_at)")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_cards_next_review ON cards(next_review)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_topic ON attempts(topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_card ON attempts(card_id)")
        logger.info("progress_store: database ready at %s", DB_PATH)
    except sqlite3.Error:
        logger.exception("progress_store: failed to initialize database")
        raise


# ---------------------------------------------------------------------------
# Answer normalization (fixes the Quiz vs Review Due mismatch permanently)
# ---------------------------------------------------------------------------
def normalize_mcq_letter(raw_answer: Any, options: Optional[List[str]] = None) -> str:
    """Coerces whatever the LLM (or a form field) gave us into a single
    clean uppercase letter (A-D) that is guaranteed to correspond to one
    of `options`, when options are supplied.

    Handles every format an LLM tends to produce:
        "A", "a", " A ", "A)", "A) Paris", "Option A", "(A)", "Answer: A"

    If the raw text doesn't look like a letter reference at all, falls
    back to matching it against the option *text* (case/whitespace
    insensitive), so a model that answers with the full option instead
    of a letter still resolves correctly.
    """
    text = str(raw_answer or "").strip()

    # Direct "A", "A)", "(A)", "Option A", "Answer: A" style matches.
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        letter = match.group(1)
        if not options or len(options) <= (ord(letter) - ord("A")):
            return letter
        return letter

    # Fall back: raw answer might be the option's full text rather than a
    # letter (e.g. "Paris" or "B) Paris" pasted without the prefix).
    if options:
        normalized_target = re.sub(r"\s+", " ", text).strip().lower()
        for idx, option in enumerate(options):
            option_clean = re.sub(r"^[A-Da-d][\)\.\:]?\s*", "", str(option)).strip()
            option_clean_norm = re.sub(r"\s+", " ", option_clean).strip().lower()
            if normalized_target and (
                normalized_target == option_clean_norm
                or normalized_target == re.sub(r"\s+", " ", str(option)).strip().lower()
            ):
                return chr(ord("A") + idx)

    # Last resort: first character if it happens to be a valid letter.
    if text[:1].upper() in {"A", "B", "C", "D"}:
        return text[:1].upper()
    return ""


def answers_match(user_answer: Any, correct_letter: str, options: Optional[List[str]] = None) -> bool:
    """Robustly compares a user-submitted MCQ answer against the stored
    correct letter, tolerant of formatting differences such as:
        "A", "A)", "A) Machine Learning", extra spaces, case.
    Exposed for any caller that wants safer grading than a plain
    string-prefix check; app1.py's own comparison already works
    correctly as long as `correct_answer` is stored as a clean letter,
    which `add_card`/`generate_quiz` now guarantee.
    """
    user_letter = normalize_mcq_letter(user_answer, options)
    return bool(user_letter) and bool(correct_letter) and user_letter == correct_letter.strip().upper()[:1]


# ---------------------------------------------------------------------------
# Card creation
# ---------------------------------------------------------------------------
def add_card(
    topic: str,
    question_type: str,
    question: str,
    options: Optional[List[str]],
    correct_answer: Optional[str],
    explanation: str = "",
    source_document: Optional[str] = None,
) -> Optional[int]:
    """Stores a freshly generated quiz question as a reviewable card.

    For MCQ cards, `correct_answer` is normalized to a single clean
    uppercase letter before being persisted (this is the permanent fix
    for "Quiz Generator marks correct, Review Due marks incorrect" —
    both screens compare against the same clean letter now). The full
    text of the correct option is also stored separately for reference.

    Returns the new card's id, or None if the write failed (never
    raises, so a persistence hiccup can't crash the quiz flow).
    """
    options = options or []
    now = datetime.datetime.now().isoformat()

    correct_letter_or_text = correct_answer or ""
    correct_option_text = ""
    if question_type == "MCQ" and options:
        letter = normalize_mcq_letter(correct_answer, options)
        if letter:
            correct_letter_or_text = letter
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(options):
                correct_option_text = str(options[idx])

    try:
        with _connect() as conn:
            cur = conn.execute(
                """INSERT INTO cards
                   (topic, question_type, question, options, correct_answer,
                    explanation, created_at, next_review, correct_option_text,
                    source_document)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    topic,
                    question_type,
                    question,
                    json.dumps(options, ensure_ascii=False),
                    correct_letter_or_text,
                    explanation or "",
                    now,
                    now,  # due immediately until first attempt
                    correct_option_text,
                    source_document or "",
                ),
            )
            return cur.lastrowid
    except sqlite3.Error:
        logger.exception("progress_store: failed to add card for topic %r", topic)
        return None


# ---------------------------------------------------------------------------
# SM-2 spaced repetition
# ---------------------------------------------------------------------------
def _sm2_update(ease_factor: float, interval: int, repetitions: int, quality: int) -> tuple:
    """Classic SM-2 algorithm. quality is 0-5 (we map correct->5, incorrect->2)."""
    if quality < 3:
        repetitions = 0
        interval = 1
    else:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 6
        else:
            interval = round(interval * ease_factor)
        repetitions += 1

    ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease_factor = max(1.3, ease_factor)
    return ease_factor, interval, repetitions


def record_attempt(card_id: int, is_correct: bool) -> None:
    """Grades an attempt, updates the card's SM-2 state, and logs it for
    the mastery dashboard. Never raises — a logging failure shouldn't
    break the quiz/review UI."""
    if card_id is None:
        return
    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
            if row is None:
                logger.warning("progress_store: record_attempt called with unknown card_id=%s", card_id)
                return

            quality = 5 if is_correct else 2
            ease, interval, reps = _sm2_update(
                row["ease_factor"], row["interval"], row["repetitions"], quality
            )
            next_review = (
                datetime.datetime.now() + datetime.timedelta(days=interval)
            ).isoformat()

            conn.execute(
                """UPDATE cards
                   SET ease_factor = ?, interval = ?, repetitions = ?, next_review = ?, last_result = ?
                   WHERE id = ?""",
                (ease, interval, reps, next_review, "correct" if is_correct else "incorrect", card_id),
            )
            conn.execute(
                "INSERT INTO attempts (card_id, topic, is_correct, answered_at) VALUES (?, ?, ?, ?)",
                (card_id, row["topic"], 1 if is_correct else 0, datetime.datetime.now().isoformat()),
            )
    except sqlite3.Error:
        logger.exception("progress_store: failed to record attempt for card_id=%s", card_id)


def get_due_cards(limit: int = 20) -> List[Dict[str, Any]]:
    """Returns cards whose next_review date has passed, oldest-due first."""
    try:
        with _connect() as conn:
            now = datetime.datetime.now().isoformat()
            rows = conn.execute(
                "SELECT * FROM cards WHERE next_review <= ? ORDER BY next_review ASC LIMIT ?",
                (now, limit),
            ).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error:
        logger.exception("progress_store: failed to fetch due cards")
        return []


def get_due_count() -> int:
    try:
        with _connect() as conn:
            now = datetime.datetime.now().isoformat()
            return conn.execute(
                "SELECT COUNT(*) as c FROM cards WHERE next_review <= ?", (now,)
            ).fetchone()["c"]
    except sqlite3.Error:
        logger.exception("progress_store: failed to count due cards")
        return 0


# ---------------------------------------------------------------------------
# Topic mastery dashboard
# ---------------------------------------------------------------------------
def get_topic_mastery() -> List[Dict[str, Any]]:
    """Aggregates all logged attempts per topic into an accuracy breakdown.
    Unchanged shape: [{"topic", "correct", "total", "accuracy"}, ...]
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT topic, SUM(is_correct) as correct, COUNT(*) as total
                   FROM attempts GROUP BY topic ORDER BY topic ASC"""
            ).fetchall()
    except sqlite3.Error:
        logger.exception("progress_store: failed to compute topic mastery")
        return []

    result = []
    for r in rows:
        total = r["total"] or 0
        correct = r["correct"] or 0
        accuracy = round(100 * correct / total, 1) if total else 0.0
        result.append(
            {"topic": r["topic"], "correct": correct, "total": total, "accuracy": accuracy}
        )
    return result


def get_dashboard_summary() -> Dict[str, Any]:
    """Extended mastery analytics beyond the per-topic table: overall
    accuracy, strong/weak topic lists, mastered/due counts, and average
    quiz score. Additive — not required by the current app1.py, but
    ready for a dashboard section to consume directly.
    """
    mastery = get_topic_mastery()
    total_correct = sum(t["correct"] for t in mastery)
    total_attempts = sum(t["total"] for t in mastery)
    overall_accuracy = round(100 * total_correct / total_attempts, 1) if total_attempts else 0.0

    strong_topics = [t["topic"] for t in mastery if t["total"] >= 3 and t["accuracy"] >= 75]
    weak_topics = [t["topic"] for t in mastery if t["total"] >= 1 and t["accuracy"] < 60]

    try:
        with _connect() as conn:
            mastered = conn.execute(
                "SELECT COUNT(*) as c FROM cards WHERE repetitions >= ? AND last_result = 'correct'",
                (MASTERED_REPETITIONS_THRESHOLD,),
            ).fetchone()["c"]
            total_cards = conn.execute("SELECT COUNT(*) as c FROM cards").fetchone()["c"]
            history_rows = conn.execute(
                """SELECT DATE(answered_at) as day, SUM(is_correct) as correct, COUNT(*) as total
                   FROM attempts GROUP BY DATE(answered_at) ORDER BY day ASC"""
            ).fetchall()
    except sqlite3.Error:
        logger.exception("progress_store: failed to compute dashboard summary")
        mastered, total_cards, history_rows = 0, 0, []

    review_history = [
        {
            "day": r["day"],
            "correct": r["correct"] or 0,
            "total": r["total"] or 0,
            "accuracy": round(100 * (r["correct"] or 0) / r["total"], 1) if r["total"] else 0.0,
        }
        for r in history_rows
    ]

    return {
        "overall_accuracy": overall_accuracy,
        "total_attempts": total_attempts,
        "correct_answers": total_correct,
        "incorrect_answers": total_attempts - total_correct,
        "questions_attempted": total_attempts,
        "questions_mastered": mastered,
        "questions_due": get_due_count(),
        "total_cards": total_cards,
        "strong_topics": strong_topics,
        "weak_topics": weak_topics,
        "review_history": review_history,
        "topic_breakdown": mastery,
    }


def reset_progress() -> None:
    """Wipes all quiz/progress data. Used alongside 'Reset Knowledge Base'."""
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM attempts")
            conn.execute("DELETE FROM cards")
        logger.info("progress_store: progress data reset")
    except sqlite3.Error:
        logger.exception("progress_store: failed to reset progress")


# ---------------------------------------------------------------------------
# Viva attempts (Confidence Analysis + Viva Analytics + Mastery integration)
# ---------------------------------------------------------------------------
def add_viva_attempt(
    difficulty: str,
    question: str,
    answer: str,
    evaluation: str,
    score: int,
    confidence: int = 0,
    accuracy: int = 0,
    completeness: int = 0,
    communication: int = 0,
) -> None:
    """Persists one graded viva answer, independent of the ephemeral
    persist/viva_progress.json file (which gets wiped on 'Start New Viva').
    This is what powers the Viva Analytics page and the Mastery Dashboard's
    viva section across sessions. Never raises."""
    try:
        with _connect() as conn:
            conn.execute(
                """INSERT INTO viva_attempts
                   (difficulty, question, answer, evaluation, score,
                    confidence, accuracy, completeness, communication, answered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    difficulty,
                    question,
                    answer,
                    evaluation,
                    int(score or 0),
                    int(confidence or 0),
                    int(accuracy or 0),
                    int(completeness or 0),
                    int(communication or 0),
                    datetime.datetime.now().isoformat(),
                ),
            )
    except sqlite3.Error:
        logger.exception("progress_store: failed to record viva attempt")


def get_viva_summary() -> Dict[str, Any]:
    """Aggregate stats across every viva attempt ever logged: average/high/
    low score, confidence-dimension averages, and the most recent viva
    date. Returns zeroed defaults if nothing has been logged yet — never
    raises, so it's always safe for a dashboard/analytics page to call."""
    empty = {
        "questions_answered": 0,
        "average_score": 0.0,
        "highest_score": 0,
        "lowest_score": 0,
        "average_confidence": 0.0,
        "average_accuracy": 0.0,
        "average_completeness": 0.0,
        "average_communication": 0.0,
        "last_viva_date": None,
    }
    try:
        with _connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as n,
                          AVG(score) as avg_score, MAX(score) as max_score, MIN(score) as min_score,
                          AVG(confidence) as avg_conf, AVG(accuracy) as avg_acc,
                          AVG(completeness) as avg_comp, AVG(communication) as avg_comm,
                          MAX(answered_at) as last_date
                   FROM viva_attempts"""
            ).fetchone()
    except sqlite3.Error:
        logger.exception("progress_store: failed to compute viva summary")
        return empty

    if not row or not row["n"]:
        return empty

    return {
        "questions_answered": row["n"],
        "average_score": round(row["avg_score"] or 0.0, 1),
        "highest_score": row["max_score"] or 0,
        "lowest_score": row["min_score"] or 0,
        "average_confidence": round(row["avg_conf"] or 0.0, 1),
        "average_accuracy": round(row["avg_acc"] or 0.0, 1),
        "average_completeness": round(row["avg_comp"] or 0.0, 1),
        "average_communication": round(row["avg_comm"] or 0.0, 1),
        "last_viva_date": row["last_date"],
    }


def get_viva_review_status() -> Optional[Dict[str, Any]]:
    """
    Determine when the user should next review viva material, based on
    their average viva score (stored 0-10, read here as a 0-100 scale).

    Rules (Task 4):
        < 40      -> Review Tomorrow
        40 - 70   -> Review in 3 Days
        70 - 90   -> Review Next Week
        > 90      -> No Review Needed

    Returns None if no viva has been taken yet, so the caller can show an
    "generate a viva first" message instead of a review card.
    """
    summary = get_viva_summary()
    if not summary["questions_answered"]:
        return None

    percent = round((summary["average_score"] / 10.0) * 100, 1)

    if percent < 40:
        days, label, urgency = 1, "Review Tomorrow", "high"
    elif percent < 70:
        days, label, urgency = 3, "Review in 3 Days", "medium"
    elif percent <= 90:
        days, label, urgency = 7, "Review Next Week", "low"
    else:
        days, label, urgency = None, "No Review Needed", "none"

    review_date = None
    if days is not None:
        review_date = (datetime.datetime.now() + datetime.timedelta(days=days)).strftime("%Y-%m-%d")

    return {
        "average_percent": percent,
        "label": label,
        "review_date": review_date,
        "days": days,
        "urgency": urgency,
    }


# ---------------------------------------------------------------------------
# Viva Analytics tab support (read-only aggregations over viva_attempts --
# no new storage; everything here reuses the existing table populated by
# progress_store.add_viva_attempt()).
# ---------------------------------------------------------------------------
def get_viva_difficulty_breakdown() -> List[Dict[str, Any]]:
    """Average viva score per difficulty level.

    The viva_attempts table doesn't tag individual questions with a topic
    (only a difficulty: Beginner/Intermediate/Advanced), so this is the
    closest existing dimension to group by. The Viva Analytics tab uses
    the best/worst entries here as "Strongest Topic" / "Weakest Topic".
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT difficulty, COUNT(*) as n, AVG(score) as avg_score
                   FROM viva_attempts GROUP BY difficulty ORDER BY difficulty ASC"""
            ).fetchall()
    except sqlite3.Error:
        logger.exception("progress_store: failed to compute viva difficulty breakdown")
        return []

    return [
        {
            "difficulty": r["difficulty"] or "Unspecified",
            "attempts": r["n"] or 0,
            "average_score": round(r["avg_score"] or 0.0, 1),
        }
        for r in rows
    ]


def get_recent_viva_attempts(limit: int = 10) -> List[Dict[str, Any]]:
    """Most recent viva attempts, newest first -- powers the Viva Analytics
    tab's 'Recent Viva Table'. Never raises."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT difficulty, question, score, confidence, answered_at
                   FROM viva_attempts ORDER BY answered_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error:
        logger.exception("progress_store: failed to fetch recent viva attempts")
        return []


def get_viva_trend() -> List[Dict[str, Any]]:
    """Full viva score + confidence history, oldest first, for the Score
    Trend / Confidence Trend charts on the Viva Analytics tab. Never
    raises -- returns [] if nothing has been logged yet."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT answered_at, score, confidence, difficulty
                   FROM viva_attempts ORDER BY answered_at ASC"""
            ).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error:
        logger.exception("progress_store: failed to compute viva trend")
        return []


def normalize_mcq_letter(correct_answer, options):
    """
    Returns only the option letter (A, B, C, D) regardless of whether
    the model returned 'A', 'A)', or 'A) Option text'.
    """
    if not correct_answer:
        return ""

    answer = str(correct_answer).strip().upper()

    # Already just A/B/C/D
    if answer in ["A", "B", "C", "D"]:
        return answer

    # Starts with A), B., C:, etc.
    if answer[0] in ["A", "B", "C", "D"]:
        return answer[0]

    # Match full option text
    for option in options:
        option = str(option).strip()
        if option.upper().startswith(answer):
            return option[0].upper()

    return ""