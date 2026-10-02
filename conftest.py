"""pytest auto-discovery hook: load .env before tests collect.

Live tests gated by env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY,
TELEGRAM_BOT_TOKEN) need the .env file loaded into os.environ to find them.
"""

from dotenv import load_dotenv

load_dotenv()
