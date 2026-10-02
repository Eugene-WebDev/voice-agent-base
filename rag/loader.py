"""Load + chunk .md files into RAGDocuments.

Token-aware chunking via tiktoken (cl100k_base — matches OpenAI embedding tokenizer).
Chunks slide with overlap so context boundaries don't drop mid-sentence.
"""

from __future__ import annotations

from pathlib import Path

import tiktoken

from core.interfaces import RAGDocument

_ENCODING_NAME = "cl100k_base"


def load_md_folder(
    folder: Path | str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[RAGDocument]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Corpus folder not found: {folder}")
    encoding = tiktoken.get_encoding(_ENCODING_NAME)

    docs: list[RAGDocument] = []
    for md_path in sorted(folder.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        rel_path = md_path.relative_to(folder)
        chunks = _chunk_text(text, chunk_size, chunk_overlap, encoding)
        for i, chunk in enumerate(chunks):
            docs.append(
                RAGDocument(
                    text=chunk,
                    source=str(rel_path),
                    metadata={"chunk_index": i, "total_chunks": len(chunks)},
                )
            )
    return docs


def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    tokens = encoding.encode(text)
    if not tokens:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - chunk_overlap
    return chunks
