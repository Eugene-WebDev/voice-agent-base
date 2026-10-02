from __future__ import annotations

import json
import time
from pathlib import Path

import aiosqlite

from .interfaces import LLMMessage, ToolCall

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    persona TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_call_id TEXT,
    tool_calls TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


class SessionStore:
    def __init__(self, db_path: Path | str = "voice_agent.db"):
        self.db_path = Path(db_path)

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def ensure_session(self, session_id: str, user_id: str, persona: str) -> None:
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO sessions (id, user_id, persona, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
                (session_id, user_id, persona, now, now),
            )
            await db.commit()

    async def append(self, session_id: str, message: LLMMessage) -> None:
        tool_calls_json = (
            json.dumps([tc.__dict__ for tc in message.tool_calls])
            if message.tool_calls
            else None
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO messages
                   (session_id, role, content, tool_call_id, tool_calls, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    message.role,
                    message.content,
                    message.tool_call_id,
                    tool_calls_json,
                    time.time(),
                ),
            )
            await db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (time.time(), session_id),
            )
            await db.commit()

    async def history(self, session_id: str, *, limit: int = 50) -> list[LLMMessage]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT role, content, tool_call_id, tool_calls FROM messages
                   WHERE session_id = ? ORDER BY id DESC LIMIT ?""",
                (session_id, limit),
            )
            rows = await cursor.fetchall()
        out: list[LLMMessage] = []
        for role, content, tool_call_id, tool_calls_json in reversed(rows):
            tool_calls = (
                [ToolCall(**tc) for tc in json.loads(tool_calls_json)]
                if tool_calls_json
                else []
            )
            out.append(
                LLMMessage(
                    role=role,
                    content=content,
                    tool_call_id=tool_call_id,
                    tool_calls=tool_calls,
                )
            )
        return out
