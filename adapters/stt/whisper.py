"""OpenAI Whisper STT adapter.

Batch-oriented: send complete audio file, get text back. Good for record-then-process
voice CRM (V1.1). For conversational <3s loops, use a streaming STT (e.g. Deepgram).
"""

from __future__ import annotations

import io
from typing import Any

from openai import AsyncOpenAI

from core.interfaces import STTProvider
from core.registry import stt_registry


@stt_registry.register("whisper")
class WhisperSTT(STTProvider):
    def __init__(
        self,
        *,
        model: str = "whisper-1",
        api_key: str | None = None,
        filename_hint: str = "audio.ogg",
        **_kwargs: Any,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        # Telegram voice notes are OGG/Opus by default; OpenAI's API needs a
        # named file-like object to infer the format.
        self.filename_hint = filename_hint

    async def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        file = io.BytesIO(audio)
        file.name = self.filename_hint

        kwargs: dict[str, Any] = {"file": file, "model": self.model}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        response = await self.client.audio.transcriptions.create(**kwargs)
        return response.text
