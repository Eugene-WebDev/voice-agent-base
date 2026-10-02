"""In-memory RAG store using numpy cosine similarity.

Sized for solo-pro corpora (~5-500 .md files, ~50-5000 chunks). For larger
corpora or persistence across restarts, swap to a chromadb-backed store —
chromadb is already an optional dep.

Lazy-indexes from `corpus_path` on first retrieve if not yet indexed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.interfaces import RAGChunk, RAGDocument, RAGStore
from core.registry import rag_registry

from .embeddings import OpenAIEmbedder
from .loader import load_md_folder


@rag_registry.register("inmemory")
class InMemoryRAGStore(RAGStore):
    def __init__(
        self,
        *,
        corpus_path: Path | str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "text-embedding-3-large",
        embedder: OpenAIEmbedder | None = None,
        **_kwargs: Any,
    ):
        self.corpus_path = Path(corpus_path) if corpus_path else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder or OpenAIEmbedder(model=embedding_model)

        self._vectors: np.ndarray | None = None
        self._docs: list[RAGDocument] = []

    @property
    def indexed(self) -> bool:
        return self._vectors is not None and len(self._docs) > 0

    async def index(self, documents: list[RAGDocument]) -> None:
        if not documents:
            self._vectors = None
            self._docs = []
            return
        texts = [d.text for d in documents]
        embeddings = await self.embedder.embed(texts)
        mat = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._vectors = mat / np.maximum(norms, 1e-9)
        self._docs = list(documents)

    async def retrieve(self, query: str, *, k: int = 5) -> list[RAGChunk]:
        if not self.indexed:
            if not self.corpus_path:
                return []
            docs = load_md_folder(
                self.corpus_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            await self.index(docs)
            if not self.indexed:
                return []

        assert self._vectors is not None
        query_emb = await self.embedder.embed([query])
        q = np.asarray(query_emb[0], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)

        scores = self._vectors @ q  # cosine similarity (vectors are L2-normalized)
        top_k = min(k, len(self._docs))
        top_indices = np.argsort(-scores)[:top_k]

        return [
            RAGChunk(
                text=self._docs[i].text,
                source=self._docs[i].source,
                score=float(scores[i]),
                metadata=self._docs[i].metadata,
            )
            for i in top_indices
        ]
