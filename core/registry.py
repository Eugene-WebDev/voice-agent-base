from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Decorator-based registry for provider classes."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self.kind} provider already registered: {name}")
            self._items[name] = cls
            return cls
        return deco

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(
                f"{self.kind} provider not registered: {name}. "
                f"Available: {sorted(self._items)}"
            )
        return self._items[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)


llm_registry = Registry("llm")
stt_registry = Registry("stt")
tts_registry = Registry("tts")
interface_registry = Registry("interface")
rag_registry = Registry("rag")
