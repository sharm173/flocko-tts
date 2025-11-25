# Multi-stage build for TTS service
# Note: Python 3.10 base image - pkuseg may fail but chatterbox-tts should work for Hindi/English
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

# Install chatterbox-tts without pkuseg (pkuseg is optional, only for Chinese)
# pkuseg fails to build on Python 3.10, but chatterbox-tts works fine without it for Hindi/English
RUN pip install --no-cache-dir --no-deps "chatterbox-tts>=0.1.4" && \
    pip install --no-cache-dir \
        "librosa==0.11.0" \
        "s3tokenizer" \
        "torch==2.6.0" \
        "torchaudio==2.6.0" \
        "transformers>=4.47.0" \
        "diffusers==0.29.0" \
        "resemble-perth==1.0.1" \
        "conformer==0.3.2" \
        "safetensors==0.5.3" && \
    python -c "import chatterbox.mtl_tts; print('✅ chatterbox-tts verified')" && \
    echo "✅ chatterbox-tts installed (pkuseg skipped - not needed for Hindi/English)"

# Production stage
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /tmp && \
    chown -R appuser:appuser /app /tmp

WORKDIR /app

# Copy installed packages
COPY --from=builder /opt/conda /opt/conda
# Create .local directory (packages are in /opt/conda, but some tools may expect .local)
RUN mkdir -p /home/appuser/.local

# Install runtime system deps (non-interactive)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg tzdata && \
    rm -rf /var/lib/apt/lists/*

# Copy application
# Note: Docker uses file checksums for COPY cache, so any change to app.py will invalidate this layer
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
