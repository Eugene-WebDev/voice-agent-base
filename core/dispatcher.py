"""Orchestration loop.

Flow:
    IncomingMessage
      → (optional) STT
      → SessionStore: append user message + fetch history
      → tool-call loop:
          LLM.generate (with system prompt, tool specs)
          if tool_calls → dispatch each tool, append assistant + tool messages, re-loop
          else → break
      → (optional) RAG retrieval injection  [Phase 3]
      → (optional) TTS
      → OutgoingMessage

Max tool-loop iterations: 5 (prevents runaway chains).
"""

from __future__ import annotations

from .config import PersonaConfig
from .interfaces import (
    IncomingMessage,
    LLMMessage,
    OutgoingMessage,
)
from .logging import SessionLogger
from .registry import (
    interface_registry,
    llm_registry,
    rag_registry,
    stt_registry,
    tts_registry,
)
from .session import SessionStore

MAX_TOOL_ITERATIONS = 5


class Dispatcher:
    def __init__(self, persona: PersonaConfig, *, db_path: str | None = None):
        self.persona = persona
        self.session_store = SessionStore(db_path or "data/voice_agent.db")

        self.llm = llm_registry.create(
            persona.llm.name,
            model=persona.llm.model,
            **persona.llm.options,
        )
        self.stt = (
            stt_registry.create(persona.stt.name, **persona.stt.options)
            if persona.stt
            else None
        )
        self.tts = (
            tts_registry.create(persona.tts.name, **persona.tts.options)
            if persona.tts
            else None
        )
        self.rag = (
            rag_registry.create(persona.rag.name, **persona.rag.options)
            if persona.rag
            else None
        )
        self.interface = interface_registry.create(
            persona.interface.name,
            **persona.interface.options,
        )

        # Tools enabled for this persona. Resolved at construction so a
        # mistyped tool name fails fast, not on first user message.
        from tools.registry import Tool, get_tool

        self._tools = [get_tool(name) for name in (persona.tools or [])]

        # RAG retrieval is wrapped as a per-persona tool bound to this dispatcher's
        # RAG store instance, so the LLM can call retrieve_context(query) when it
        # needs SOP knowledge. Not registered globally because the binding is
        # persona-specific.
        if self.rag and persona.rag:
            rag_store = self.rag
            top_k = persona.rag.top_k

            async def retrieve_context(query: str) -> str:
                """Search the knowledge base for relevant context.

                Use when the user asks about procedures, policies, or domain
                knowledge that isn't in the conversation history. Returns top
                matching .md excerpts with their source paths.
                """
                chunks = await rag_store.retrieve(query, k=top_k)
                if not chunks:
                    return "No relevant context found in the knowledge base."
                return "\n\n---\n\n".join(
                    f"[{c.source}]\n{c.text}" for c in chunks
                )

            self._tools.append(
                Tool.from_function(retrieve_context, name="retrieve_context")
            )

        self._tools_by_name = {t.name: t for t in self._tools}

    async def start(self) -> None:
        await self.session_store.init()
        await self.interface.start(self.handle)

    async def handle(self, msg: IncomingMessage) -> OutgoingMessage | None:
        session_id = f"session-{msg.user_id}"
        logger = SessionLogger(session_id)
        logger.log(
            "incoming",
            user_id=msg.user_id,
            has_audio=msg.audio is not None,
            text_len=len(msg.text),
        )
        await self.session_store.ensure_session(session_id, msg.user_id, self.persona.name)

        text = msg.text
        if msg.audio and self.stt:
            text = await self.stt.transcribe(msg.audio, language=self.persona.language)
            logger.log("stt_done", text=text)

        if not text:
            return None

        user_msg = LLMMessage(role="user", content=text)
        await self.session_store.append(session_id, user_msg)
        history = await self.session_store.history(session_id)

        tool_specs = [t.spec() for t in self._tools] if self._tools else None
        last_content = ""

        for iteration in range(MAX_TOOL_ITERATIONS):
            result = await self.llm.generate(
                history,
                system=self.persona.system_prompt,
                tools=tool_specs,
                max_tokens=self.persona.llm.max_tokens,
                temperature=self.persona.llm.temperature,
            )
            logger.log(
                "llm_done",
                iteration=iteration,
                content_len=len(result.content),
                usage=result.usage,
                tool_call_count=len(result.tool_calls),
            )

            assistant_msg = LLMMessage(
                role="assistant",
                content=result.content,
                tool_calls=result.tool_calls,
            )
            await self.session_store.append(session_id, assistant_msg)
            history.append(assistant_msg)
            if result.content:
                last_content = result.content

            if not result.tool_calls:
                break

            for tc in result.tool_calls:
                logger.log("tool_call", name=tc.name, arguments=tc.arguments)
                tool_fn = self._tools_by_name.get(tc.name)
                if tool_fn is None:
                    tool_result = (
                        f"Error: tool '{tc.name}' is not enabled for this persona."
                    )
                else:
                    try:
                        raw = await tool_fn(**tc.arguments)
                        tool_result = str(raw)
                    except Exception as e:  # noqa: BLE001  - tools can raise anything; report it
                        tool_result = f"Error executing {tc.name}: {e}"
                logger.log("tool_done", name=tc.name, result_len=len(tool_result))
                tool_msg = LLMMessage(
                    role="tool",
                    content=tool_result,
                    tool_call_id=tc.id,
                )
                await self.session_store.append(session_id, tool_msg)
                history.append(tool_msg)

        audio: bytes | None = None
        if self.tts and self.persona.tts and last_content:
            audio = await self.tts.synthesize(
                last_content,
                voice_id=self.persona.tts.voice_id,
                language=self.persona.language,
            )
            logger.log("tts_done", audio_bytes=len(audio))

        return OutgoingMessage(user_id=msg.user_id, text=last_content, audio=audio)


async def main() -> None:
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    import adapters  # noqa: F401  triggers provider registration
    import tools  # noqa: F401  triggers tool registration

    from .config import load_persona

    load_dotenv()
    persona_path = os.environ.get("PERSONA", "personas/sample.yaml")
    persona = load_persona(Path(persona_path))
    db_path = os.environ.get("DB_PATH", "data/voice_agent.db")
    dispatcher = Dispatcher(persona, db_path=db_path)
    await dispatcher.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
