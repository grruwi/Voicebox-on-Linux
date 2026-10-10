# Voicebox (backend + panel webowy) bez torcha. Mowa idzie z kontenera `audiocpp`,
# rozpoznawanie mowy z kontenera `whisper` — tu siedzi tylko API, panel i MCP.

FROM oven/bun:1 AS frontend
WORKDIR /build
COPY package.json bun.lock CHANGELOG.md ./
COPY app/ ./app/
COPY web/ ./web/
# workspace'y desktopowe (tauri, landing) nie wchodza do repo
RUN sed -i '/"tauri"/d; /"landing"/d' package.json && sed -i -z 's/,\n  ]/\n  ]/' package.json
RUN bun install --no-save
RUN cd web && bunx --bun vite build


FROM python:3.12-slim
# ffmpeg — konwersja probek; ffplay z niego gra na glosnikach serwera (opcjonalne, patrz compose)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-slim.txt .
RUN pip install --no-cache-dir -r requirements-slim.txt
COPY backend/ /app/backend/
COPY --from=frontend /build/web/dist /app/frontend/

# Bez tego Voicebox buduje domyslny silnik PyTorchowy (torcha w obrazie nie ma).
ENV VOICEBOX_TYLKO_VOXCPM=1 \
    VOXCPM_GGUF=1 \
    VOICEBOX_STT_ENGINE=whispercpp \
    VOXCPM_SERWER=http://audiocpp:17496 \
    AUDIOCPP_ZARZADCA=http://audiocpp:17495 \
    VOICEBOX_MODELE=/modele \
    HOME=/app/data \
    NUMBA_CACHE_DIR=/app/data/numba-cache

EXPOSE 17493
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:17493/health || exit 1
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "17493"]
