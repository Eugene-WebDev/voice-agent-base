# voice-agent-base

Reusable base for voice agents serving solo professionals (real-estate agents, lawyers, consultants).

**One framework, many clients.** Each client = one YAML persona + one `.md` folder + their own provider keys. The base handles Telegram I/O, STT/LLM/TTS, persona memory, tool-calling, and RAG over the `.md` knowledge corpus.

**Status:** Phases 0–4 complete. Code-complete, lint-clean, 48 mocked tests + 3 live tests passing. Production-deployable.

## Architecture

```
Telegram voice/text
        │
        ▼
   Dispatcher  ──► Session memory (SQLite, per-user)
        │
        ▼
      Whisper STT  (audio → PL/EN text)
        │
        ▼
  Claude Sonnet 4.6  ◄── system prompt (cached, ephemeral)
        │           ◄── conversation history
        │           ◄── tools (persona-declared + RAG)
        │
        ▼
   Tool-call loop (max 5 iterations)
        │   ├── get_current_time
        │   ├── lookup_client_record
        │   ├── draft_client_email
        │   └── retrieve_context  ◄── RAG over .md corpus
        │
        ▼
  ElevenLabs TTS Turbo  (Polish-capable, multilingual)
        │
        ▼
   Telegram voice + text reply
```

Adapter pattern: each layer is swappable. Whisper → Deepgram, Claude → Gemini/OpenAI, ElevenLabs → other TTS, Telegram → Slack/WhatsApp/CLI — all without touching `core/` or `tools/`.

## Quickstart

```bash
# 1. Set up Python 3.11+ environment
uv venv --python 3.12        # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[claude,openai,elevenlabs,telegram,rag,dev]"

# 2. Fill keys
cp .env.example .env
# Edit .env with ANTHROPIC_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY, TELEGRAM_BOT_TOKEN

# 3. Verify
pytest tests/unit -v          # 48 mocked tests should pass
pytest tests/unit -m live -v  # opt-in live API smoke tests (uses real APIs, ~$0.01)

# 4. Run
python -m core.dispatcher     # starts the bot, polls Telegram
```

Send a voice note from Telegram in Polish; receive PL text + audio reply within ~8s.

## Add a new client (onboarding checklist, target <4h)

1. **Persona file** — copy `personas/sample-rag.yaml` → `personas/<client>.yaml`. Edit:
   - `name`, `language`, `system_prompt` (their domain, tone, register)
   - `llm.model` (Sonnet 4.6 default; Haiku 4.5 for cheaper/faster)
   - `tts.voice_id` (browse [elevenlabs.io/app/voices](https://elevenlabs.io/app/voices) for a voice matching their language)
   - `rag.corpus_path` (points to their `.md` folder)
   - `tools` list (declare which built-in tools they get)
2. **Knowledge corpus** — create `personas/<client>_corpus/` with their SOPs, vocab, templates as `.md` files. Plain markdown. No frontmatter required.
3. **Custom tools** (if needed) — add to `tools/examples/<client>_*.py` with `@tool()` decorator. Type hints auto-generate the JSON schema for the LLM.
4. **Telegram bot** — create a fresh bot via [@BotFather](https://t.me/BotFather), copy the token.
5. **Env** — set `PERSONA=personas/<client>.yaml` and `TELEGRAM_BOT_TOKEN=<their_token>` in `.env` (or a per-client `.env.<client>`).
6. **Deploy** — `docker compose up -d` (see Deployment below).

## Persona YAML reference

```yaml
name: my-client                       # session-store key prefix
language: pl                          # default for STT/TTS
system_prompt: |                      # cached ephemerally on every LLM call
  Multi-line persona definition.

llm:
  name: claude                        # adapter registered in adapters/llm/
  model: claude-sonnet-4-6
  max_tokens: 1024
  temperature: 0.6

stt:
  name: whisper                       # adapter in adapters/stt/
  language: pl
  options:
    model: whisper-1
    # prompt: "Warsaw districts: Mokotów, Wola, Śródmieście."  # bias vocab

tts:
  name: elevenlabs
  voice_id: 21m00Tcm4TlvDq8ikWAM      # from elevenlabs.io/app/voices
  language: pl
  options:
    model_id: eleven_turbo_v2_5

rag:                                  # omit (or null) to disable RAG
  name: inmemory
  corpus_path: personas/my-client_corpus
  chunk_size: 512
  chunk_overlap: 64
  embedding_model: text-embedding-3-large
  top_k: 3

interface:
  name: telegram
  options:
    allowed_user_ids: []              # empty = open; populate to whitelist

tools:                                # declared by name (registered via @tool)
  - get_current_time
  - lookup_client_record
  - draft_client_email
  # retrieve_context is auto-added when `rag` is set
```

## Layout

```
core/                     Interfaces, config, logging, session, dispatcher
  ├── interfaces.py       5 ABCs + 8 dataclasses
  ├── config.py           Pydantic v2 persona schema + YAML loader
  ├── session.py          aiosqlite per-user message store
  ├── logging.py          JSONL session-event emitter
  ├── registry.py         Provider registries
  └── dispatcher.py       Orchestration loop (STT → tool-loop → TTS)
adapters/                 Provider implementations
  ├── llm/claude.py       Anthropic, prompt caching, tool-call shape
  ├── stt/whisper.py      OpenAI Whisper, OGG/Opus hint for Telegram
  ├── tts/elevenlabs.py   AsyncElevenLabs streaming → bytes
  └── interface/telegram.py  python-telegram-bot v21+, voice + text handlers
tools/                    LLM tool declarations
  ├── registry.py         @tool decorator + Tool.from_function
  └── examples/           get_current_time, lookup_client_record, draft_client_email
rag/                      Knowledge retrieval
  ├── loader.py           .md → token-aware chunks (tiktoken)
  ├── embeddings.py       OpenAI text-embedding-3-large wrapper
  └── inmemory_store.py   numpy cosine, lazy index from corpus_path
personas/                 Client-specific configs
  ├── sample.yaml         Tools-only sample
  ├── sample-rag.yaml     Tools + RAG sample
  └── sample_corpus/      Example .md SOPs
tests/                    pytest, unit (mocked) + live-tagged opt-in
```

## Built-in tools

| Tool | Purpose | Status |
|---|---|---|
| `get_current_time` | Current date/time in any IANA timezone | Real |
| `lookup_client_record` | CRM lookup by client_id | Phase 2 mock — wire real Airtable adapter for production |
| `draft_client_email` | Format an email draft (doesn't send) | Phase 2 mock — wire real email adapter to actually send |
| `retrieve_context` | Search the `.md` knowledge base, returns top-k chunks | Real — auto-added when `rag` is configured |

Add custom tools by dropping a Python file in `tools/examples/`:

```python
from tools.registry import tool

@tool()
def my_custom_tool(arg1: str, arg2: int = 5) -> str:
    """One-line description used by the LLM to decide when to call this tool."""
    return f"got {arg1} and {arg2}"
```

Type hints become the JSON schema; the first paragraph of the docstring is the LLM-facing description.

## Deployment

### Docker (recommended)

```bash
docker compose up -d
docker compose logs -f
```

`docker-compose.yml` mounts `./personas`, `./logs`, and `./data` as volumes so persona edits + session DB persist across container restarts.

### Hostinger VPS (or any VPS with Docker)

1. SSH to VPS, install Docker + docker compose plugin.
2. `git clone` (or `scp -r`) the project to `/opt/voice-agent-base/`.
3. Create `.env` with production keys.
4. `cd /opt/voice-agent-base && docker compose up -d`.
5. Verify with `docker compose logs -f agent`.

For per-client isolation: deploy one container per persona/bot, each with its own `.env.<client>` and Telegram bot token. Compose stacks scale horizontally.

### Costs (per active client, rough estimate)

- Anthropic Claude Sonnet 4.6: ~$0.02 per 10-message conversation
- OpenAI Whisper: $0.006/min audio
- OpenAI embeddings (RAG): one-time + occasional re-index, negligible
- ElevenLabs Turbo v2.5: ~$0.03 per 1000 chars (Starter = $5/mo, 30k chars)
- Hostinger VPS: $5–15/mo depending on tier

Single active user, light use: budget **~$10–25/month** all-in.

## Development

```bash
uv pip install -e ".[dev,claude,openai,elevenlabs,telegram,rag]"

# All mocked unit tests (no keys needed, ~5s)
pytest tests/unit -q

# Live API smoke tests (needs .env with all keys, ~$0.01 cost)
pytest tests/unit -m live -v

# Lint
ruff check .
```

## Roadmap (out of scope for current cap)

- Streaming dialogue (Deepgram STT) — for use cases needing <3s conversational latency
- Self-host Whisper / LLM — when API costs justify infra
- Real Airtable + email adapters (replace Phase 2 mocks)
- Per-client persistence layer for RAG (chromadb PersistentClient already optional)
- Voice cloning of the agent's own voice (ElevenLabs Professional Voice Cloning)
- Multi-user / multi-tenant per-bot (current = one bot, multiple users via session-per-chat-id)

Evaluated and deferred for now: PersonaPlex, Ghost.

## License

Personal R&D, not yet open-sourced. Internal use.

## Evals

`evals/` holds a regression suite for the RAG persona: retrieval hit-rate, required-fact checks, refusal on out-of-scope questions, and an LLM-as-judge groundedness score, with thresholds that fail the run. See [`evals/README.md`](evals/README.md).
