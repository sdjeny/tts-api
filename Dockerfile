# ============================================================
# TTS API — Multi-model Docker Image
# Supports: Qwen3-TTS CustomVoice / Base / Kokoro-82M
# ============================================================
FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── Application code ─────────────────────────────────────
WORKDIR /app
COPY server.py .
COPY tts_client.py .
COPY config.yaml .
COPY handlers/ ./handlers/

# ── Data volumes ─────────────────────────────────────────
# /models  — mount model files here (Qwen3-TTS / Kokoro)
# /audio   — generated audio output
# /clones  — voice clone storage (Base model)
VOLUME ["/models", "/audio", "/clones"]

RUN mkdir -p /audio /clones

# ── Runtime ──────────────────────────────────────────────
# Override config paths for container layout
ENV TTS_MODEL_PATH=/models
ENV TTS_OUTPUT_DIR=/audio
ENV TTS_CLONE_DIR=/clones
ENV TTS_CONFIG=/app/config.yaml

EXPOSE 8420

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/tts/health')" || exit 1

ENTRYPOINT ["python", "server.py"]
