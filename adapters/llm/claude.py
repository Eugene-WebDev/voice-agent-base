"""Anthropic Claude LLM adapter.

Uses the official Anthropic Python SDK. System prompt is cached via prompt-caching
(ephemeral, 5-min TTL) — per the claude-api skill, this saves ~90% on repeated calls
with the same persona. Stable system prompt → cache hit on every turn after the first.

Tool-calling is wired through but not exercised in Phase 1 (Phase 2 deliverable).
Streaming is not used — voice-agent responses are typically short and TTS-bound.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from core.interfaces import (
    LLMMessage,
    LLMProvider,
    LLMResult,
    ToolCall,
    ToolSpec,
)
from core.registry import llm_registry


@llm_registry.register("claude")
class ClaudeLLM(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        cache_system_prompt: bool = True,
        **_kwargs: Any,
    ):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.cache_system_prompt = cache_system_prompt

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._convert_messages(messages),
        }

        if system:
            kwargs["system"] = (
                [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                if self.cache_system_prompt
                else system
            )

        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        response = await self.client.messages.create(**kwargs)

        content_text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(arguments))
                )

        usage = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "cache_creation": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }

        return LLMResult(
            content=content_text,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=response.stop_reason or "end_turn",
        )

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert internal LLMMessage list to Anthropic Messages API format.

        - system messages are filtered (passed via `system` kwarg)
        - tool results (role=tool) become user-role messages with tool_result content blocks
        - assistant messages with tool_calls produce text + tool_use blocks
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content,
                            }
                        ],
                    }
                )
                continue
            if msg.role == "assistant" and msg.tool_calls:
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                continue
            out.append({"role": msg.role, "content": msg.content})
        return out
