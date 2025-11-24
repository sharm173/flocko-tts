# TTS Service - Chatterbox Multilingual TTS

TTS microservice using Chatterbox Multilingual TTS for Hindi/English voice synthesis.

## Quick Start

### Option 1: Local Testing (No Model - Fast API Testing)

```bash
# Install minimal dependencies
pip install -r requirements-local.txt

# Run in stub mode
MODEL_LOADING_ENABLED=false uvicorn app:app --reload
```

See [QUICK_START.md](QUICK_START.md) for details.

### Option 2: Full Testing with Model (Docker)

```bash
# Build and run with Docker
./run_docker.sh

# Test synthesis
./test_synthesis.sh
```

See [TEST_WITH_MODEL.md](TEST_WITH_MODEL.md) for details.

## Architecture

- **FastAPI** service with structured logging
- **Chatterbox Multilingual TTS** for speech synthesis
- **4-bit quantization** for GPU optimization
- **Health checks** (`/healthz`, `/readyz`)
- **Production-ready** Docker image

## API Endpoints

### `POST /synthesize`

Synthesize text to speech and return WAV file.

**Form Data:**
- `text` (required): Text to synthesize
- `language` (optional): Language code (default: `hi`)
- `speaker` (optional): Speaker identifier
- `speed` (optional): Speech speed (default: 1.0)

**Response:** WAV audio file

### `POST /v1/tts/stream`

Stream TTS audio in chunks.

**JSON Body:**
```json
{
  "text": "नमस्ते",
  "language": "hi",
  "speaker": "default",
  "speed": 1.0
}
```

**Response:** Streaming audio chunks

### `GET /healthz`

Liveness probe - always returns 200 if service is running.

### `GET /readyz`

Readiness probe - returns 200 when model is loaded, 503 otherwise.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGUAGE` | `hi` | Default language code |
| `USE_4BIT` | `true` | Enable 4-bit quantization (GPU only) |
| `MODEL_ID` | `chatterbox-multilingual` | Model identifier |
| `MODEL_LOADING_ENABLED` | `true` | Enable/disable model loading |
| `REQUIRE_MODEL` | `false` | Fail startup if model can't load |

## Local Development

### Prerequisites

- Python 3.12+ (for API testing without model)
- Docker (for full testing with model)

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# For API testing (no model)
pip install -r requirements-local.txt

# For full testing (use Docker)
docker build --platform linux/amd64 -t tts-service:latest .
```

## Docker Deployment

```bash
# Build
docker build --platform linux/amd64 -t tts-service:latest .

# Run
docker run -d --name tts-service \
  -p 8000:8000 \
  --gpus all \
  tts-service:latest

# Test
curl http://localhost:8000/healthz
```

## Testing

### Quick API Test (No Model)

```bash
MODEL_LOADING_ENABLED=false uvicorn app:app --reload
curl http://localhost:8000/healthz
```

### Full Test (With Model)

```bash
./run_docker.sh
./test_synthesis.sh
```

## Troubleshooting

### Python 3.12 Compatibility

`chatterbox-tts` requires `numpy<1.26.0`, but Python 3.12 only has numpy wheels for `>=1.26.0`.

**Solution**: Use Docker for full testing (works fine on linux/amd64).

### Model Loading Issues

- Check Docker logs: `docker logs tts-service`
- Verify GPU availability: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Increase Docker memory limit (8GB+ recommended)

### Audio Quality

- Verify language parameter matches text
- Check model loaded correctly in logs
- Try different text samples

## Production Deployment

See Kubernetes manifests in `../k8s/tts-deploy.yaml` for production deployment.

## License

Proprietary - IndiGo Airlines
# flocko-tts
