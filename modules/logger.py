from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

from config import settings


def log_query(
    query: str,
    answer: str,
    namespace: str,
    top_k: int,
    score_threshold: float,
    sources: Optional[List[dict]] = None,
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None,
):
    os.makedirs(os.path.dirname(settings.log_path) or ".", exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer,
        "namespace": namespace,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "sources": sources or [],
    }
    with open(settings.log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_recent_logs(n: int = 20) -> List[dict]:
    if not os.path.exists(settings.log_path):
        return []
    with open(settings.log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    return [json.loads(line) for line in lines if line.strip()]


def log_file_path() -> str:
    return settings.log_path


def clear_logs() -> None:
    """Deletes the persisted query log file. Used by the Settings > Logs
    'Clear logs' action."""
    if os.path.exists(settings.log_path):
        os.remove(settings.log_path)
