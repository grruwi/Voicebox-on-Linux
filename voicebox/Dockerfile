# Voicebox — the backend + web panel + MCP, without torch. Speech comes from the engine
# containers (`audiocpp`, optionally `pytorch`), speech recognition from `whisper`.

FROM oven/bun:1 AS frontend
WORKDIR /build
COPY package.json bun.lock CHANGELOG.md ./
COPY app/ ./app/
COPY web/ ./web/
# the desktop workspaces (tauri, landing) are not part of this repo
RUN sed -i '/"tauri"/d; /"landing"/d' package.json && sed -i -z 's/,\n  ]/\n  ]/' package.json
RUN bun install --no-save
RUN cd web && bunx --bun vite build


FROM python:3.12-slim
# ffmpeg — sample conversion; its ffplay plays on the server's speakers (optional, see compose)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-slim.txt .
RUN pip install --no-cache-dir -r requirements-slim.txt
COPY backend/ /app/backend/
COPY --from=frontend /build/web/dist /app/frontend/

# Engine addresses are the compose service names; models are mounted at /models.
ENV VOICEBOX_VOICE_MODELS=/models \
    HOME=/app/data \
    HF_HOME=/app/data/hf \
    NUMBA_CACHE_DIR=/app/data/numba-cache \
    PYTHONUNBUFFERED=1

EXPOSE 17493
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:17493/health || exit 1
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "17493"]
