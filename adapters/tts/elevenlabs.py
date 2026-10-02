"""ElevenLabs TTS adapter.

Uses ElevenLabs' Python SDK (v1+) with the AsyncElevenLabs client.
`text_to_speech.convert(...)` returns an async iterator of audio bytes; we
accumulate to a single bytes payload for downstream Telegram voice-reply.
"""

from __future__ import annotations

import os
from typing import Any

from elevenlabs.client import AsyncElevenLabs

from core.interfaces import TTSProvider
from core.registry import tts_registry


@tts_registry.register("elevenlabs")
class ElevenLabsTTS(TTSProvider):
    def __init__(
        self,
        *,
        model_id: str = "eleven_turbo_v2_5",
        api_key: str | None = None,
        output_format: str = "mp3_44100_128",
        **_kwargs: Any,
    ):
        # ElevenLabs SDK doesn't fall back to env when api_key=None is passed
        # explicitly — resolve here so persona.tts.options can omit the key
        # and rely on ELEVENLABS_API_KEY env var.
        resolved_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ElevenLabs API key not provided. Set ELEVENLABS_API_KEY env "
                "or pass api_key in persona.tts.options."
            )
        self.client = AsyncElevenLabs(api_key=resolved_key)
        self.model_id = model_id
        self.output_format = output_format

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        language: str | None = None,
        format: str = "mp3",
    ) -> bytes:
        # language is informational on ElevenLabs — handled via the chosen
        # voice/model_id, not a separate parameter. Kept in the signature
        # for interface symmetry.
        stream = self.client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=self.model_id,
            output_format=self.output_format,
        )
        chunks: list[bytes] = []
        async for chunk in stream:
            chunks.append(chunk)
        return b"".join(chunks)
