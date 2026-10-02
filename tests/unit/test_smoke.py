"""Phase 0 smoke tests: imports work, interfaces are abstract, config loads, registry behaves."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from core import (
    InterfaceProvider,
    LLMMessage,
    LLMProvider,
    PersonaConfig,
    RAGStore,
    SessionLogger,
    STTProvider,
    ToolCall,
    TTSProvider,
    llm_registry,
    load_persona,
    stt_registry,
)
from core.registry import Registry


def test_interfaces_are_abstract():
    for cls in (LLMProvider, STTProvider, TTSProvider, InterfaceProvider, RAGStore):
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]


def test_sample_persona_loads():
    persona = load_persona("personas/sample.yaml")
    assert isinstance(persona, PersonaConfig)
    assert persona.name == "sample"
    assert persona.llm.name == "claude"
    assert persona.stt is not None and persona.stt.name == "whisper"
    assert persona.tts is not None and persona.tts.voice_id


def test_persona_validation_rejects_missing_required():
    with pytest.raises(ValidationError):
        PersonaConfig.model_validate({"name": "x"})


def test_registry_register_and_create():
    reg = Registry("llm")

    class FakeLLM(LLMProvider):
        async def generate(
            self,
            messages,
            *,
            system=None,
            tools=None,
            max_tokens=1024,
            temperature=0.7,
        ):
            return None  # type: ignore[return-value]

    reg.register("fake")(FakeLLM)
    assert "fake" in reg.names()
    inst = reg.create("fake")
    assert isinstance(inst, FakeLLM)

    with pytest.raises(ValueError):
        reg.register("fake")(FakeLLM)
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_global_registries_exist():
    assert llm_registry.kind == "llm"
    assert stt_registry.kind == "stt"


def test_llm_message_with_tool_calls():
    msg = LLMMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="t1", name="ping", arguments={"x": 1})],
    )
    assert msg.tool_calls[0].name == "ping"


def test_session_logger_writes_jsonl(tmp_path: Path):
    logger = SessionLogger("test-sess", log_dir=tmp_path)
    logger.log("hello", x=1, text="żółć")
    logger.log("done")

    files = list(tmp_path.rglob("test-sess.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["event"] == "hello"
    assert rec["session_id"] == "test-sess"
    assert rec["x"] == 1
    assert rec["text"] == "żółć"  # diacritics preserved


@pytest.mark.asyncio
async def test_session_store_round_trip(tmp_path: Path):
    from core.session import SessionStore

    store = SessionStore(tmp_path / "test.db")
    await store.init()
    await store.ensure_session("s1", "user-a", "sample")
    await store.append("s1", LLMMessage(role="user", content="cześć"))
    await store.append("s1", LLMMessage(role="assistant", content="hej"))
    history = await store.history("s1")
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].content == "cześć"  # diacritics preserved through SQLite


def test_yaml_persona_round_trips():
    raw = yaml.safe_load(Path("personas/sample.yaml").read_text(encoding="utf-8"))
    persona = PersonaConfig.model_validate(raw)
    assert persona.language == "pl"
