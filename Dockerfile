# Multi-stage build for TTS service
# Using Python 3.9 base image for pkuseg compatibility
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel as builder

WORKDIR /app

# Install system dependencies (non-interactive)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg build-essential tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Use requirements-docker.txt if it exists, otherwise requirements.txt
COPY requirements*.txt /app/

# Install numpy and build tools first
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "numpy>=1.24.0,<1.26.0"

# Install all dependencies except chatterbox-tts
RUN if [ -f requirements-docker.txt ]; then \
        grep -v "chatterbox-tts" requirements-docker.txt > /tmp/reqs_no_chatterbox.txt && \
        pip install --no-cache-dir -r /tmp/reqs_no_chatterbox.txt; \
    else \
        grep -v "chatterbox-tts" requirements.txt > /tmp/reqs_no_chatterbox.txt && \
        pip install --no-cache-dir -r /tmp/reqs_no_chatterbox.txt; \
    fi

# Try to install pkuseg (optional - for Chinese, not needed for Hindi/English)
# If it fails, that's OK - we'll install chatterbox without it
RUN (pip install --no-cache-dir --no-build-isolation "pkuseg==0.0.25" && echo "✅ pkuseg installed") || \
    echo "⚠️ pkuseg failed (will continue without Chinese segmentation support)"

# Install chatterbox-tts
# If pkuseg failed, chatterbox may still work for Hindi/English
RUN pip install --no-cache-dir "chatterbox-tts>=0.1.4" && echo "✅ chatterbox-tts installed"

# Production stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /tmp && \
    chown -R appuser:appuser /app /tmp

WORKDIR /app

# Copy installed packages
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.local /home/appuser/.local

# Install runtime system deps (non-interactive)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg tzdata && \
    rm -rf /var/lib/apt/lists/*

# Copy application
COPY --chown=appuser:appuser app.py /app/

ENV PATH=/home/appuser/.local/bin:/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TORCH_AUDIO_USE_SOUNDFILE=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/healthz', timeout=2)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
