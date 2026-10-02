from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LLMMessage:
    role: Literal["user", "assistant", "system", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class LLMResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResult: ...


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str: ...


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        language: str | None = None,
        format: str = "mp3",
    ) -> bytes: ...


@dataclass
class IncomingMessage:
    user_id: str
    text: str = ""
    audio: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    user_id: str
    text: str = ""
    audio: bytes | None = None


MessageHandler = Callable[[IncomingMessage], Awaitable["OutgoingMessage | None"]]


class InterfaceProvider(ABC):
    @abstractmethod
    async def start(self, handler: MessageHandler) -> None: ...

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> None: ...


@dataclass
class RAGDocument:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGChunk:
    text: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGStore(ABC):
    @abstractmethod
    async def index(self, documents: list[RAGDocument]) -> None: ...

    @abstractmethod
    async def retrieve(self, query: str, *, k: int = 5) -> list[RAGChunk]: ...
