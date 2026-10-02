"""Tool declaration DSL + registry.

A `@tool()` decorator wraps a Python function into a `Tool` object:
    - name        ← function name (overridable)
    - description ← first paragraph of docstring
    - parameters  ← JSON schema inferred from type hints + defaults

The dispatcher passes `Tool.spec()` to the LLM (as `ToolSpec`) and dispatches
`Tool.__call__(**arguments)` when the LLM returns a matching tool_call.

Phase 2: simple types (str/int/float/bool). Phase 3+ may extend to list/dict.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from core.interfaces import ToolSpec


class Tool:
    def __init__(
        self,
        fn: Callable,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ):
        self.fn = fn
        self.name = name
        self.description = description
        self.parameters = parameters

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    async def __call__(self, **kwargs: Any) -> Any:
        if asyncio.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        return self.fn(**kwargs)

    @classmethod
    def from_function(cls, fn: Callable, *, name: str | None = None) -> Tool:
        """Build a Tool from a function without registering it globally.

        Used for persona-scoped tools (e.g., RAG retrieval bound to a specific
        store instance) that shouldn't pollute the global tool registry.
        """
        return cls(
            fn=fn,
            name=name or fn.__name__,
            description=_extract_description(fn),
            parameters=_signature_to_json_schema(fn),
        )


_tools: dict[str, Tool] = {}


def tool(name: str | None = None) -> Callable[[Callable], Tool]:
    def deco(fn: Callable) -> Tool:
        actual_name = name or fn.__name__
        if actual_name in _tools:
            raise ValueError(f"Tool already registered: {actual_name}")
        t = Tool(
            fn=fn,
            name=actual_name,
            description=_extract_description(fn),
            parameters=_signature_to_json_schema(fn),
        )
        _tools[actual_name] = t
        return t

    return deco


def get_tool(name: str) -> Tool:
    if name not in _tools:
        raise KeyError(
            f"Tool not registered: {name}. Available: {sorted(_tools)}"
        )
    return _tools[name]


def all_tools() -> dict[str, Tool]:
    return dict(_tools)


def _extract_description(fn: Callable) -> str:
    doc = inspect.getdoc(fn) or ""
    if not doc:
        return f"Tool {fn.__name__}"
    return doc.split("\n\n")[0].strip()


_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _signature_to_json_schema(fn: Callable) -> dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        py_type = hints.get(param_name, str)
        json_type = _TYPE_MAP.get(py_type, "string")
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
