"""Phase 2 tool-calling tests.

Covers:
- @tool decorator: name, description, JSON schema from type hints + defaults
- Tool execution: sync and async functions
- Registry: register, get, duplicate-rejection, missing-name
- Sample tools behave (get_current_time / lookup_client_record / draft_client_email)
- Dispatcher tool-loop: LLM tool_call → dispatch → re-prompt → final reply
"""

from __future__ import annotations

import pytest

import adapters  # noqa: F401  ensure provider registry populated for dispatcher test

# Alias the tools-package import to avoid shadowing the `tools=` parameter
# in fake LLM `generate` methods below (LLMProvider interface signature).
import tools as _tools_loaded  # noqa: F401
from core import (
    IncomingMessage,
    InterfaceConfig,
    LLMConfig,
    LLMResult,
    PersonaConfig,
    ToolCall,
)
from core.dispatcher import Dispatcher
from core.registry import interface_registry, llm_registry
from tools.registry import Tool, all_tools, get_tool, tool

# ---------- @tool decorator ----------


def test_tool_decorator_infers_name_description_and_schema():
    @tool(name="example_unique_1")
    def my_fn(a: str, b: int = 5) -> str:
        """Do something useful.

        Longer details that should not appear in description.
        """
        return f"{a}:{b}"

    assert my_fn.name == "example_unique_1"
    assert my_fn.description == "Do something useful."
    assert my_fn.parameters["type"] == "object"
    assert my_fn.parameters["properties"]["a"] == {"type": "string"}
    assert my_fn.parameters["properties"]["b"] == {"type": "integer"}
    assert "a" in my_fn.parameters["required"]
    assert "b" not in my_fn.parameters["required"]


def test_tool_decorator_defaults_to_function_name():
    @tool()
    def example_unique_2(x: bool) -> bool:
        """Echo a bool."""
        return x

    assert example_unique_2.name == "example_unique_2"
    assert example_unique_2.parameters["properties"]["x"] == {"type": "boolean"}


def test_tool_decorator_with_float_param():
    @tool()
    def example_unique_3(price: float) -> float:
        """Echo a price."""
        return price

    assert example_unique_3.parameters["properties"]["price"] == {"type": "number"}


@pytest.mark.asyncio
async def test_tool_call_sync_function():
    @tool()
    def example_unique_sync(n: int) -> int:
        """Double it."""
        return n * 2

    result = await example_unique_sync(n=21)
    assert result == 42


@pytest.mark.asyncio
async def test_tool_call_async_function():
    @tool()
    async def example_unique_async(n: int) -> int:
        """Triple it."""
        return n * 3

    result = await example_unique_async(n=7)
    assert result == 21


# ---------- registry ----------


def test_registry_rejects_duplicate():
    @tool(name="dup_test_unique")
    def first() -> None:
        """First."""

    with pytest.raises(ValueError):

        @tool(name="dup_test_unique")
        def second() -> None:  # noqa: F841
            """Second."""


def test_registry_get_missing_raises():
    with pytest.raises(KeyError):
        get_tool("definitely-not-a-tool")


def test_all_tools_includes_examples():
    names = set(all_tools())
    assert "get_current_time" in names
    assert "lookup_client_record" in names
    assert "draft_client_email" in names


# ---------- sample tools behavior ----------


@pytest.mark.asyncio
async def test_get_current_time_returns_iso_like_string():
    t = get_tool("get_current_time")
    out = await t(timezone="Europe/Warsaw")
    assert isinstance(out, str)
    assert len(out) > 0
    # Year prefix sanity check
    assert out[:4].isdigit()


@pytest.mark.asyncio
async def test_lookup_client_record_finds_known():
    t = get_tool("lookup_client_record")
    out = await t(client_id="adam-kowalski")
    assert "Adam Kowalski" in out
    assert "Mokotów" in out


@pytest.mark.asyncio
async def test_lookup_client_record_missing_id():
    t = get_tool("lookup_client_record")
    out = await t(client_id="nonexistent")
    assert "No client found" in out


@pytest.mark.asyncio
async def test_draft_client_email_formats_output():
    t = get_tool("draft_client_email")
    out = await t(
        client_name="Adam Kowalski",
        subject="Mieszkanie na Mokotowie",
        body="W załączeniu szczegóły oferty.",
    )
    assert "Adam Kowalski" in out
    assert "NOT SENT" in out


# ---------- dispatcher tool-loop integration ----------


@pytest.mark.asyncio
async def test_dispatcher_executes_tool_and_returns_final_reply(tmp_path):
    """LLM returns tool_call → dispatcher invokes tool → re-prompts LLM → returns final."""

    call_log: list[dict] = []

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
                    "messages_count": len(messages),
                    "tools_provided": tools is not None and len(tools) > 0,
                }
            )
            if len(call_log) == 1:
                # First call: ask for the time
                return LLMResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="get_current_time",
                            arguments={"timezone": "Europe/Warsaw"},
                        )
                    ],
                    usage={"input": 10, "output": 5},
                    finish_reason="tool_use",
                )
            # Second call: produce a final reply using the tool result
            last = messages[-1]
            tool_result_text = last.content if last.role == "tool" else ""
            return LLMResult(
                content=f"Czas: {tool_result_text}",
                tool_calls=[],
                usage={"input": 30, "output": 10},
                finish_reason="end_turn",
            )

    class FakeInterface:
        def __init__(self, **_kwargs):
            self.started = False

        async def start(self, handler):
            self.started = True

        async def send(self, message):
            pass

    # Register fakes under unique names; clean up after test.
    llm_registry._items["fake-llm-tooltest"] = FakeLLM
    interface_registry._items["fake-iface-tooltest"] = FakeInterface
    try:
        persona = PersonaConfig(
            name="test-tool-loop",
            system_prompt="Jesteś asystentem.",
            language="pl",
            llm=LLMConfig(
                name="fake-llm-tooltest",
                model="any",
                max_tokens=200,
                temperature=0.5,
            ),
            stt=None,
            tts=None,
            rag=None,
            interface=InterfaceConfig(name="fake-iface-tooltest"),
            tools=["get_current_time"],
        )
        dispatcher = Dispatcher(persona, db_path=str(tmp_path / "test.db"))
        await dispatcher.session_store.init()

        result = await dispatcher.handle(
            IncomingMessage(user_id="u1", text="która godzina?")
        )

        assert result is not None
        # LLM called twice: once with tools, once after tool result
        assert len(call_log) == 2
        assert call_log[0]["tools_provided"] is True
        assert call_log[1]["tools_provided"] is True
        # Final reply text incorporates the tool's stringified output (year prefix)
        assert "Czas:" in result.text
        assert result.text[6:10].strip().isdigit() or "2026" in result.text
    finally:
        del llm_registry._items["fake-llm-tooltest"]
        del interface_registry._items["fake-iface-tooltest"]


@pytest.mark.asyncio
async def test_dispatcher_handles_unknown_tool_gracefully(tmp_path):
    """LLM hallucinates a tool name → dispatcher returns error in tool_result, doesn't crash."""

    call_count = [0]

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

        async def generate(self, messages, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return LLMResult(
                    content="",
                    tool_calls=[
                        ToolCall(id="t1", name="not_a_real_tool", arguments={})
                    ],
                    usage={"input": 5, "output": 1},
                )
            # Second call sees error and recovers
            return LLMResult(
                content="Sorry, can't do that.",
                tool_calls=[],
                usage={"input": 10, "output": 5},
            )

    class FakeInterface:
        def __init__(self, **_kwargs):
            pass

        async def start(self, handler):
            pass

        async def send(self, message):
            pass

    llm_registry._items["fake-llm-unknown-tool"] = FakeLLM
    interface_registry._items["fake-iface-unknown-tool"] = FakeInterface
    try:
        persona = PersonaConfig(
            name="t",
            system_prompt="x",
            language="en",
            llm=LLMConfig(name="fake-llm-unknown-tool", model="any"),
            interface=InterfaceConfig(name="fake-iface-unknown-tool"),
            tools=["get_current_time"],
        )
        dispatcher = Dispatcher(persona, db_path=str(tmp_path / "unk.db"))
        await dispatcher.session_store.init()
        result = await dispatcher.handle(IncomingMessage(user_id="u1", text="hi"))
        assert result is not None
        assert "Sorry" in result.text
        # Tool result message recorded in history
        history = await dispatcher.session_store.history("session-u1")
        tool_msgs = [m for m in history if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert "not enabled" in tool_msgs[0].content
    finally:
        del llm_registry._items["fake-llm-unknown-tool"]
        del interface_registry._items["fake-iface-unknown-tool"]


# ---------- persona validation ----------


def test_dispatcher_raises_on_unknown_persona_tool():
    """Mistyped tool name in persona.tools fails fast at Dispatcher construction."""

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

    class FakeInterface:
        def __init__(self, **_kwargs):
            pass

    llm_registry._items["fake-llm-mis"] = FakeLLM
    interface_registry._items["fake-iface-mis"] = FakeInterface
    try:
        persona = PersonaConfig(
            name="t",
            system_prompt="x",
            language="en",
            llm=LLMConfig(name="fake-llm-mis", model="any"),
            interface=InterfaceConfig(name="fake-iface-mis"),
            tools=["definitely_not_a_real_tool_name"],
        )
        with pytest.raises(KeyError):
            Dispatcher(persona, db_path="/tmp/never-created.db")
    finally:
        del llm_registry._items["fake-llm-mis"]
        del interface_registry._items["fake-iface-mis"]


# Make Tool import available for symmetry in IDE imports (silences F401 thinkers)
_ = Tool
