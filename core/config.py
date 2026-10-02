from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    name: str
    options: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(ProviderConfig):
    model: str
    max_tokens: int = 1024
    temperature: float = 0.7


class STTConfig(ProviderConfig):
    language: str | None = None


class TTSConfig(ProviderConfig):
    voice_id: str
    language: str | None = None


class RAGConfig(ProviderConfig):
    corpus_path: Path
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "text-embedding-3-large"
    top_k: int = 5


class InterfaceConfig(ProviderConfig):
    pass


class PersonaConfig(BaseModel):
    name: str
    system_prompt: str
    language: str = "en"
    llm: LLMConfig
    stt: STTConfig | None = None
    tts: TTSConfig | None = None
    rag: RAGConfig | None = None
    interface: InterfaceConfig
    tools: list[str] = Field(default_factory=list)


def load_persona(path: Path | str) -> PersonaConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Persona file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Persona file must be a YAML mapping: {p}")
    return PersonaConfig.model_validate(data)
