FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY core ./core
COPY adapters ./adapters
COPY tools ./tools
COPY rag ./rag

RUN pip install --upgrade pip && pip install ".[claude,openai,gemini,elevenlabs,telegram,rag]"

COPY personas ./personas

CMD ["python", "-m", "core.dispatcher"]
