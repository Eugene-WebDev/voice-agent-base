"""OpenAI embedding wrapper.

Default model: text-embedding-3-large (3072-dim). For Polish corpora, performance
is acceptable out of the box; if retrieval quality is poor, swap to a Polish-native
embedding (sdadas/mmlw-roberta-large via sentence-transformers) — left as a plugin
point.
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-large",
        api_key: str | None = None,
    ):
        resolved = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved:
            raise ValueError(
                "OpenAI API key not provided for embeddings. "
                "Set OPENAI_API_KEY or pass api_key."
            )
        self.client = AsyncOpenAI(api_key=resolved)
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
