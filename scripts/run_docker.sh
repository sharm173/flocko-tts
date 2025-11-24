#!/bin/bash
# Quick script to run TTS service in Docker with model

set -e

cd "$(dirname "$0")"

IMAGE_NAME="tts-service:latest"
CONTAINER_NAME="tts-service"
PORT="${PORT:-8000}"

# Check if image exists
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "📦 Building Docker image (this may take 10-15 minutes)..."
    docker build --platform linux/amd64 -t "$IMAGE_NAME" .
fi

# Stop existing container if running
if docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "🛑 Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Run container
echo "🚀 Starting TTS service container..."
echo "   Image: $IMAGE_NAME"
echo "   Port: $PORT"
echo "   Container: $CONTAINER_NAME"
echo ""

# Try GPU first, fallback to CPU
if docker run --gpus all --help >/dev/null 2>&1; then
    echo "   Using GPU support"
    docker run -d --name "$CONTAINER_NAME" \
        -p "$PORT:8000" \
        --gpus all \
        "$IMAGE_NAME"
else
    echo "   Using CPU (GPU not available)"
    docker run -d --name "$CONTAINER_NAME" \
        -p "$PORT:8000" \
        "$IMAGE_NAME"
fi

echo ""
echo "⏳ Waiting for service to start..."
sleep 3

echo ""
echo "📋 Container logs (Ctrl+C to exit):"
echo "   Use 'docker logs -f $CONTAINER_NAME' to follow logs"
echo ""

# Wait for model to load
echo "⏳ Waiting for model to load (this may take 1-2 minutes)..."
for i in {1..60}; do
    STATUS=$(curl -s "http://localhost:$PORT/readyz" 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "ready" ]; then
        echo ""
        echo "✅ Service is ready!"
        echo ""
        echo "🧪 Test it:"
        echo "   curl http://localhost:$PORT/healthz"
        echo "   ./test_synthesis.sh http://localhost:$PORT"
        echo ""
        echo "📊 View logs:"
        echo "   docker logs -f $CONTAINER_NAME"
        echo ""
        echo "🛑 Stop service:"
        echo "   docker stop $CONTAINER_NAME"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "⚠️  Service may still be loading. Check logs:"
echo "   docker logs $CONTAINER_NAME"

