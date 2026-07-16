import json
import os
from rag_engine1 import generate_learning_path as llm_generate_learning_path
LEARNING_PATH_FILE = os.path.join("persist", "learning_path.json")


def ensure_storage():
    os.makedirs("persist", exist_ok=True)


def save_learning_path(path):
    ensure_storage()
    with open(LEARNING_PATH_FILE, "w", encoding="utf-8") as f:
        json.dump(path, f, indent=2, ensure_ascii=False)


def load_learning_path():
    ensure_storage()

    if not os.path.exists(LEARNING_PATH_FILE):
        return []

    with open(LEARNING_PATH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def mark_completed(index):
    path = load_learning_path()

    if 0 <= index < len(path):
        path[index]["completed"] = True

    save_learning_path(path)


def mark_pending(index):
    path = load_learning_path()

    if 0 <= index < len(path):
        path[index]["completed"] = False

    save_learning_path(path)


def learning_progress():
    path = load_learning_path()

    if len(path) == 0:
        return 0

    completed = sum(1 for item in path if item["completed"])

    return int((completed / len(path)) * 100)

def reset_learning_path():
    save_learning_path([])