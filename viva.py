import asyncio
import io
import json
import os

import edge_tts
import speech_recognition as sr
from pydub import AudioSegment


VIVA_FILE = os.path.join("persist", "viva_progress.json")

# Voice used for question narration. Any edge-tts neural voice name works;
# run `edge-tts --list-voices` to see the full catalogue.
VIVA_TTS_VOICE = "en-US-AriaNeural"


# =====================================================
# Storage
# =====================================================

def ensure_storage():
    os.makedirs("persist", exist_ok=True)


def save_progress(data):
    ensure_storage()

    with open(VIVA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_progress():

    ensure_storage()

    if not os.path.exists(VIVA_FILE):
        return {
            "questions": [],
            "current": 0,
            "score": 0,
            "history": [],
            "completed": False
        }

    with open(VIVA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def reset_progress():

    save_progress(
        {
            "questions": [],
            "current": 0,
            "score": 0,
            "history": [],
            "completed": False
        }
    )


# =====================================================
# Text To Speech  (edge-tts, browser playback via st.audio)
# =====================================================

async def _synthesize_speech(text, voice=VIVA_TTS_VOICE):
    """Stream mp3 audio bytes for `text` from edge-tts."""

    communicate = edge_tts.Communicate(text, voice)

    audio_bytes = b""

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    return audio_bytes


def speak(text, voice=VIVA_TTS_VOICE):
    """
    Synthesize `text` to speech and return raw mp3 bytes.

    Replaces the old pyttsx3 implementation, which spoke through the
    server's local speakers (only works when Streamlit runs on the same
    machine as the user). This version returns audio bytes so the caller
    can play them in-browser with st.audio(...), which works for any
    deployment (local or hosted).
    """

    if not text or not text.strip():
        return b""

    try:
        return asyncio.run(_synthesize_speech(text, voice))
    except RuntimeError:
        # asyncio.run() fails if an event loop is already running
        # (can happen depending on how Streamlit schedules callbacks).
        # Fall back to a fresh loop in that case.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_synthesize_speech(text, voice))
        finally:
            loop.close()


# =====================================================
# Speech To Text  (browser mic recording via streamlit-mic-recorder)
# =====================================================

def listen(audio_bytes):
    """
    Transcribe recorded browser audio to text.

    Replaces the old speech_recognition-with-local-microphone implementation
    (via sr.Microphone()), which only worked when the Streamlit process had
    direct access to a local mic. Now the microphone lives in the browser:
    the caller records audio client-side with streamlit-mic-recorder and
    passes the raw recorded bytes (webm/wav) in here.

    Returns an empty string if nothing could be transcribed, rather than
    raising, so the caller can show a friendly "please try again" message.
    """

    if not audio_bytes:
        return ""

    try:
        # streamlit-mic-recorder typically returns webm/opus bytes; convert
        # to PCM WAV in-memory so speech_recognition can read it. Requires
        # ffmpeg to be installed and on PATH (pydub shells out to it).
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        return recognizer.recognize_google(audio_data)

    except sr.UnknownValueError:
        # Audio was decoded fine but no speech could be understood.
        return ""
    except Exception:
        # Any decoding/network error - fail soft, let the UI ask to retry.
        return ""


# =====================================================
# Question Navigation
# =====================================================

def current_question():

    data = load_progress()

    if data["current"] >= len(data["questions"]):
        return None

    return data["questions"][data["current"]]


def next_question():

    data = load_progress()

    data["current"] += 1

    if data["current"] >= len(data["questions"]):
        data["completed"] = True

    save_progress(data)


# =====================================================
# Save Evaluation
# =====================================================

def add_feedback(question, answer, evaluation):

    data = load_progress()

    data["history"].append(
        {
            "question": question,
            "answer": answer,
            "evaluation": evaluation
        }
    )

    save_progress(data)


def add_confidence(confidence):
    """
    Attach a confidence-analysis breakdown (confidence / accuracy /
    completeness / communication) to the most recently recorded history
    entry. Meant to be called right after add_feedback() for the same
    answer. Purely additive -- if there's no history yet, this is a no-op,
    and it never touches add_feedback's own behavior or data shape.
    """

    data = load_progress()

    if data["history"]:
        data["history"][-1]["confidence"] = confidence
        save_progress(data)


# =====================================================
# Score
# =====================================================

def add_score(score):

    data = load_progress()

    data["score"] += score

    save_progress(data)


def viva_percentage():

    data = load_progress()

    total = len(data["questions"])

    if total == 0:
        return 0

    return round(
        (data["score"] / (total * 10)) * 100,
        1
    )