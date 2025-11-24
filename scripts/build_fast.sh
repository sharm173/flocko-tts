#!/bin/bash
# Faster Docker build with optimizations

set -e

cd "$(dirname "$0")"

echo "🚀 Building TTS service (optimized)..."
echo ""

# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with cache optimization
docker build \
  --platform linux/amd64 \
  --progress=plain \
  --tag tts-service:latest \
  .

echo ""
echo "✅ Build complete!"
echo ""
echo "To run:"
echo "  docker run -d --name tts-service -p 8000:8000 tts-service:latest"
echo "  docker logs -f tts-service"

