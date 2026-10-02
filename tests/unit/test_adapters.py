"""Phase 1 adapter tests.

Mocked tests: provider SDKs are mocked, assertion targets are the request shape
and the response-handling logic. These run on every pytest invocation.

Live tests: marked @pytest.mark.live, skipped unless the corresponding env var
is set. Run explicitly with `pytest -m live` once API keys are in .env.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import adapters  # noqa: F401  side-effect import triggers provider registration
from core import (
    LLMMessage,
    ToolCall,
    interface_registry,
    llm_registry,
    stt_registry,
    tts_registry,
)

# ---------- registration ----------


def test_all_adapters_register_in_global_registries():
    assert "claude" in llm_registry.names()
    assert "whisper" in stt_registry.names()
    assert "elevenlabs" in tts_registry.names()
    assert "telegram" in interface_registry.names()


# ---------- Claude conversion ----------


def _claude_no_init():
    """Return a ClaudeLLM instance without running __init__ (which needs an API key)."""
    from adapters.llm.claude import ClaudeLLM

    return ClaudeLLM.__new__(ClaudeLLM)


def test_claude_converts_simple_user_message():
    out = _claude_no_init()._convert_messages([LLMMessage(role="user", content="cześć")])
    assert out == [{"role": "user", "content": "cześć"}]


def test_claude_preserves_user_assistant_alternation():
    msgs = [
        LLMMessage(role="user", content="hi"),
        LLMMessage(role="assistant", content="hello"),
        LLMMessage(role="user", content="how are you?"),
    ]
    out = _claude_no_init()._convert_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]


def test_claude_converts_assistant_tool_call_and_tool_result():
    msgs = [
        LLMMessage(role="user", content="what time is it?"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="t1", name="get_time", arguments={})],
        ),
        LLMMessage(role="tool", content="12:00", tool_call_id="t1"),
    ]
    out = _claude_no_init()._convert_messages(msgs)
    assert out[1]["role"] == "assistant"
    assert out[1]["content"][0]["type"] == "tool_use"
    assert out[1]["content"][0]["id"] == "t1"
    assert out[2]["role"] == "user"
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "t1"


def test_claude_filters_system_role_messages():
    out = _claude_no_init()._convert_messages(
        [
            LLMMessage(role="system", content="ignored"),
            LLMMessage(role="user", content="hi"),
        ]
    )
    assert out == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_claude_generate_request_shape_and_caching(monkeypatch):
    from adapters.llm import claude as claude_mod
    from adapters.llm.claude import ClaudeLLM

    fake_text_block = MagicMock(type="text", text="Cześć!")
    fake_resp = MagicMock()
    fake_resp.content = [fake_text_block]
    fake_resp.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    fake_resp.stop_reason = "end_turn"

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=fake_resp)
    monkeypatch.setattr(claude_mod, "AsyncAnthropic", lambda **_kw: mock_client)

    adapter = ClaudeLLM(model="claude-sonnet-4-6")
    result = await adapter.generate(
        [LLMMessage(role="user", content="cześć")],
        system="Jesteś pomocnym asystentem.",
        max_tokens=100,
        temperature=0.6,
    )

    assert result.content == "Cześć!"
    assert result.usage["input"] == 10
    assert result.usage["output"] == 5
    assert result.finish_reason == "end_turn"

    kwargs = mock_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.6
    # System prompt cached as a list with ephemeral cache_control
    assert isinstance(kwargs["system"], list)
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"][0]["text"] == "Jesteś pomocnym asystentem."


# ---------- Whisper ----------


@pytest.mark.asyncio
async def test_whisper_transcribe_passes_language_and_model(monkeypatch):
    from adapters.stt import whisper as whisper_mod
    from adapters.stt.whisper import WhisperSTT

    fake_resp = MagicMock(text="Cześć, to test")
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = AsyncMock(return_value=fake_resp)
    monkeypatch.setattr(whisper_mod, "AsyncOpenAI", lambda **_kw: mock_client)

    adapter = WhisperSTT()
    result = await adapter.transcribe(b"fake-audio", language="pl")

    assert result == "Cześć, to test"
    kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["language"] == "pl"
    assert kwargs["file"].name == "audio.ogg"


# ---------- ElevenLabs ----------


@pytest.mark.asyncio
async def test_elevenlabs_synthesize_accumulates_stream(monkeypatch):
    from adapters.tts import elevenlabs as eleven_mod
    from adapters.tts.elevenlabs import ElevenLabsTTS

    async def fake_stream():
        yield b"chunk1-"
        yield b"chunk2"

    mock_client = MagicMock()
    mock_client.text_to_speech.convert = MagicMock(return_value=fake_stream())
    monkeypatch.setattr(eleven_mod, "AsyncElevenLabs", lambda **_kw: mock_client)

    adapter = ElevenLabsTTS(model_id="eleven_turbo_v2_5")
    result = await adapter.synthesize("Cześć", voice_id="voice123")

    assert result == b"chunk1-chunk2"
    kwargs = mock_client.text_to_speech.convert.call_args.kwargs
    assert kwargs["voice_id"] == "voice123"
    assert kwargs["model_id"] == "eleven_turbo_v2_5"
    assert kwargs["text"] == "Cześć"


# ---------- Telegram ----------


def test_telegram_requires_token(monkeypatch):
    from adapters.interface.telegram import TelegramInterface

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(ValueError):
        TelegramInterface()


def test_telegram_accepts_explicit_token():
    from adapters.interface.telegram import TelegramInterface

    adapter = TelegramInterface(token="dummy:token")
    assert adapter.token == "dummy:token"


def test_telegram_reads_token_from_env(monkeypatch):
    from adapters.interface.telegram import TelegramInterface

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env:token")
    adapter = TelegramInterface()
    assert adapter.token == "env:token"


def test_telegram_allowed_user_filter():
    from adapters.interface.telegram import TelegramInterface

    open_bot = TelegramInterface(token="t")
    locked_bot = TelegramInterface(token="t", allowed_user_ids=[42])
    assert open_bot._is_allowed(1) is True
    assert locked_bot._is_allowed(42) is True
    assert locked_bot._is_allowed(99) is False


# ---------- live tests (opt-in via -m live; need real keys) ----------


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_claude_generate_pl():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from adapters.llm.claude import ClaudeLLM

    adapter = ClaudeLLM(model="claude-sonnet-4-6")
    result = await adapter.generate(
        [LLMMessage(role="user", content="Powiedz tylko jedno słowo: cześć.")],
        system="Jesteś zwięzłym asystentem. Odpowiadasz po polsku.",
        max_tokens=20,
    )
    assert result.content
    assert result.usage["input"] > 0


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_whisper_transcribe_pl():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    pytest.skip("Requires a real PL audio file at tests/fixtures/sample_pl.ogg")


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_elevenlabs_pl_voice():
    if not os.environ.get("ELEVENLABS_API_KEY"):
        pytest.skip("ELEVENLABS_API_KEY not set")
    from adapters.tts.elevenlabs import ElevenLabsTTS

    adapter = ElevenLabsTTS()
    audio = await adapter.synthesize(
        "Cześć, to jest test polskiego głosu.",
        voice_id="21m00Tcm4TlvDq8ikWAM",
    )
    assert len(audio) > 1000
