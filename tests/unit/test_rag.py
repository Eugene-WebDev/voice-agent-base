"""Phase 3 RAG tests.

Covers:
- Loader: chunks .md files token-aware, attaches source + index metadata
- Embedder: mocked OpenAI client, batch handling, empty input
- InMemoryRAGStore: index, retrieve top-k, cosine ordering, lazy load from corpus_path
- Dispatcher: builds retrieve_context tool when persona.rag is set
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import adapters  # noqa: F401  ensure provider registry populated
import rag  # noqa: F401  ensure rag store registered
import tools as _tools_loaded  # noqa: F401
from core import (
    IncomingMessage,
    InterfaceConfig,
    LLMConfig,
    LLMResult,
    PersonaConfig,
    RAGConfig,
    ToolCall,
)
from core.dispatcher import Dispatcher
from core.registry import interface_registry, llm_registry
from rag.embeddings import OpenAIEmbedder
from rag.inmemory_store import InMemoryRAGStore
from rag.loader import load_md_folder

# ---------- loader ----------


def test_loader_reads_md_and_chunks(tmp_path: Path):
    (tmp_path / "a.md").write_text("Paragraph A.\n\nParagraph B." * 50, encoding="utf-8")
    (tmp_path / "b.md").write_text("Short doc.", encoding="utf-8")
    docs = load_md_folder(tmp_path, chunk_size=64, chunk_overlap=8)
    assert len(docs) >= 2
    assert all(d.source.endswith(".md") for d in docs)
    sources = {d.source for d in docs}
    assert "a.md" in sources
    assert "b.md" in sources
    a_chunks = [d for d in docs if d.source == "a.md"]
    assert len(a_chunks) > 1  # long doc must split
    assert a_chunks[0].metadata["chunk_index"] == 0
    assert a_chunks[-1].metadata["total_chunks"] == len(a_chunks)


def test_loader_preserves_pl_diacritics(tmp_path: Path):
    (tmp_path / "pl.md").write_text("Wspólnota mieszkaniowa i spółdzielnia.", encoding="utf-8")
    docs = load_md_folder(tmp_path, chunk_size=64, chunk_overlap=8)
    assert any("Wspólnota" in d.text for d in docs)
    assert any("spółdzielnia" in d.text for d in docs)


def test_loader_raises_on_missing_folder(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_md_folder(tmp_path / "nonexistent")


def test_loader_rejects_overlap_ge_size(tmp_path: Path):
    (tmp_path / "x.md").write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        load_md_folder(tmp_path, chunk_size=64, chunk_overlap=64)


# ---------- embedder ----------


@pytest.mark.asyncio
async def test_embedder_calls_openai_with_model_and_input(monkeypatch):
    from rag import embeddings as emb_mod

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(
        return_value=MagicMock(
            data=[
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.4, 0.5, 0.6]),
            ]
        )
    )
    monkeypatch.setattr(emb_mod, "AsyncOpenAI", lambda **_kw: mock_client)

    embedder = OpenAIEmbedder(model="text-embedding-3-large")
    vectors = await embedder.embed(["doc 1", "doc 2"])
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    kwargs = mock_client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-large"
    assert kwargs["input"] == ["doc 1", "doc 2"]


@pytest.mark.asyncio
async def test_embedder_empty_input_skips_api():
    """Avoid spurious API calls on empty input."""
    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.client = MagicMock()  # never used
    embedder.model = "x"
    result = await embedder.embed([])
    assert result == []
    embedder.client.embeddings.create.assert_not_called()


# ---------- in-memory store ----------


class _FakeEmbedder:
    """Deterministic embedder for tests — maps simple texts to known vectors."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(t, [0.0, 0.0, 0.0]) for t in texts]


@pytest.mark.asyncio
async def test_store_index_and_retrieve_orders_by_similarity():
    from core.interfaces import RAGDocument

    vectors = {
        "viewing procedure for clients": [1.0, 0.0, 0.0],
        "followup after viewing": [0.9, 0.1, 0.0],
        "unrelated text about pasta recipes": [0.0, 1.0, 0.0],
        "How do I follow up?": [0.95, 0.05, 0.0],
    }
    store = InMemoryRAGStore(embedder=_FakeEmbedder(vectors))
    docs = [
        RAGDocument(text="viewing procedure for clients", source="sop-viewing.md"),
        RAGDocument(text="followup after viewing", source="sop-followup.md"),
        RAGDocument(text="unrelated text about pasta recipes", source="recipes.md"),
    ]
    await store.index(docs)
    assert store.indexed

    chunks = await store.retrieve("How do I follow up?", k=2)
    assert len(chunks) == 2
    # Both relevant docs should rank above the pasta one
    sources = [c.source for c in chunks]
    assert "recipes.md" not in sources
    # Scores should be in descending order
    assert chunks[0].score >= chunks[1].score


@pytest.mark.asyncio
async def test_store_lazy_loads_from_corpus_path_on_first_retrieve(tmp_path: Path):
    (tmp_path / "doc.md").write_text("Some text about real estate procedures.", encoding="utf-8")
    vectors = {
        "Some text about real estate procedures.": [1.0, 0.0, 0.0],
        "real estate query": [0.99, 0.01, 0.0],
    }
    store = InMemoryRAGStore(
        corpus_path=tmp_path,
        embedder=_FakeEmbedder(vectors),
    )
    assert not store.indexed
    chunks = await store.retrieve("real estate query", k=1)
    assert store.indexed
    assert len(chunks) == 1
    assert chunks[0].source == "doc.md"


@pytest.mark.asyncio
async def test_store_empty_corpus_returns_empty_list(tmp_path: Path):
    store = InMemoryRAGStore(
        corpus_path=tmp_path,
        embedder=_FakeEmbedder({}),
    )
    chunks = await store.retrieve("anything", k=5)
    assert chunks == []


# ---------- dispatcher RAG tool wiring ----------


@pytest.mark.asyncio
async def test_dispatcher_creates_retrieve_context_tool_when_rag_configured(tmp_path: Path):
    """When persona.rag is set, dispatcher should expose a retrieve_context tool to the LLM."""

    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "sop.md").write_text("RAG procedure here.", encoding="utf-8")

    call_log: list[dict] = []
    rag_chunks_returned: list[list] = []

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

        async def generate(
            self,
            messages,
            *,
            system=None,
            tools=None,
            max_tokens=1024,
            temperature=0.7,
        ):
            call_log.append(
                {
                    "tool_names": [t.name for t in (tools or [])],
                    "messages_count": len(messages),
                }
            )
            if len(call_log) == 1:
                return LLMResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="r1",
                            name="retrieve_context",
                            arguments={"query": "RAG procedure"},
                        )
                    ],
                    usage={"input": 5, "output": 2},
                )
            last_tool = messages[-1].content if messages[-1].role == "tool" else ""
            rag_chunks_returned.append([last_tool])
            return LLMResult(
                content=f"Per SOP: {last_tool[:50]}",
                tool_calls=[],
                usage={"input": 20, "output": 10},
            )

    class FakeInterface:
        def __init__(self, **_kwargs):
            pass

        async def start(self, handler):
            pass

        async def send(self, message):
            pass

    # Inject a real InMemoryRAGStore with our fake embedder so we don't hit OpenAI.
    class _StubEmbedder:
        async def embed(self, texts):
            return [[1.0] + [0.0] * 9 for _ in texts]

    real_store = InMemoryRAGStore(
        corpus_path=tmp_path / "corpus",
        embedder=_StubEmbedder(),
    )

    llm_registry._items["fake-llm-rag"] = FakeLLM
    interface_registry._items["fake-iface-rag"] = FakeInterface
    from core.registry import rag_registry

    rag_registry._items["fake-rag"] = lambda **_kwargs: real_store
    try:
        persona = PersonaConfig(
            name="t",
            system_prompt="x",
            language="pl",
            llm=LLMConfig(name="fake-llm-rag", model="any"),
            rag=RAGConfig(
                name="fake-rag",
                corpus_path=tmp_path / "corpus",
                chunk_size=64,
                chunk_overlap=8,
                top_k=2,
            ),
            interface=InterfaceConfig(name="fake-iface-rag"),
            tools=[],
        )
        dispatcher = Dispatcher(persona, db_path=str(tmp_path / "rag.db"))
        await dispatcher.session_store.init()

        result = await dispatcher.handle(IncomingMessage(user_id="u1", text="jaka procedura?"))

        assert result is not None
        assert "Per SOP" in result.text
        # Tool list on first call must include retrieve_context
        assert "retrieve_context" in call_log[0]["tool_names"]
        # Tool was actually executed (returned chunks from the corpus)
        assert rag_chunks_returned
        assert "RAG procedure here." in rag_chunks_returned[0][0]
    finally:
        del llm_registry._items["fake-llm-rag"]
        del interface_registry._items["fake-iface-rag"]
        del rag_registry._items["fake-rag"]
