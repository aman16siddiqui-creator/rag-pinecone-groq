from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

CONV_DIR = "data/conversations"
SETTINGS_PATH = "data/app_settings.json"


def _ensure_dirs() -> None:
    os.makedirs(CONV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SETTINGS_PATH) or ".", exist_ok=True)


def _path(conv_id: str) -> str:
    return os.path.join(CONV_DIR, f"{conv_id}.json")


def new_conversation() -> dict:
    """Creates a fresh, empty, not-yet-saved conversation record with
    its own dedicated namespace."""
    conv_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": conv_id,
        "title": "New chat",
        "created_at": now,
        "updated_at": now,
        "namespace": f"chat-{conv_id}",
        "processed_docs": [],
        "messages": [],
    }


def save_conversation(conv: dict) -> None:
    """Overwrites the conversation's file with its current in-memory
    state. Called after every message so nothing is ever lost."""
    _ensure_dirs()
    conv = dict(conv)
    conv["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(_path(conv["id"]), "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)


def load_conversation(conv_id: str) -> Optional[dict]:
    path = _path(conv_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def list_conversations() -> List[dict]:
    """Lightweight summaries (id, title, updated_at) of every non-empty
    saved conversation, newest first."""
    _ensure_dirs()
    summaries = []
    for fname in os.listdir(CONV_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(CONV_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("messages"):
            continue  # don't show chats nothing ever happened in
        summaries.append({
            "id": data["id"],
            "title": data.get("title") or "New chat",
            "updated_at": data.get("updated_at", data.get("created_at", "")),
        })
    summaries.sort(key=lambda s: s["updated_at"], reverse=True)
    return summaries


def load_settings() -> dict:
    _ensure_dirs()
    if not os.path.exists(SETTINGS_PATH):
        return {"theme": "system"}
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"theme": "system"}


def save_settings(settings_dict: dict) -> None:
    _ensure_dirs()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, ensure_ascii=False, indent=2)
