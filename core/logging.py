from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SessionLogger:
    """Per-session JSONL logger. One file per session_id under logs/YYYY-MM-DD/."""

    def __init__(self, session_id: str, log_dir: Path | str = "logs"):
        self.session_id = session_id
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        self.dir = Path(log_dir) / date
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{session_id}.jsonl"

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": time.time(),
            "session_id": self.session_id,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
