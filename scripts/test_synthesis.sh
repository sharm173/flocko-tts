#!/bin/bash
# Test script for TTS synthesis

set -e

SERVICE_URL="${1:-http://localhost:8000}"

echo "🧪 Testing TTS Service at $SERVICE_URL"
echo ""

# Check if service is reachable
if ! curl -s "$SERVICE_URL/healthz" >/dev/null 2>&1; then
    echo "❌ Service not reachable at $SERVICE_URL"
    echo "   Make sure the service is running:"
    echo "   - Docker: ./run_docker.sh"
    echo "   - Local: MODEL_LOADING_ENABLED=false uvicorn app:app --reload"
    exit 1
fi

# Test health
echo "1. Testing health endpoint..."
curl -s "$SERVICE_URL/healthz" | python3 -m json.tool
echo ""

# Test readiness
echo "2. Testing readiness endpoint..."
curl -s "$SERVICE_URL/readyz" | python3 -m json.tool
echo ""

# Wait for model to load if needed
echo "3. Waiting for model to be ready..."
for i in {1..30}; do
    STATUS=$(curl -s "$SERVICE_URL/readyz" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "ready" ]; then
        echo "   ✅ Model is ready!"
        break
    fi
    echo "   ⏳ Waiting... ($i/30)"
    sleep 2
done
echo ""

# Test synthesis
echo "4. Testing TTS synthesis (Hindi)..."
curl -X POST "$SERVICE_URL/synthesize" \
  -F "text=नमस्ते, मैं IndiGo की सहायता कर रहा हूं। आपका PNR क्या है?" \
  -F "language=hi" \
  -o output_hindi.wav

if [ -f output_hindi.wav ]; then
    SIZE=$(ls -lh output_hindi.wav | awk '{print $5}')
    echo "   ✅ Generated output_hindi.wav ($SIZE)"
    echo "   Play with: afplay output_hindi.wav (macOS) or aplay output_hindi.wav (Linux)"
else
    echo "   ❌ Failed to generate audio"
fi
echo ""

# Test English
echo "5. Testing TTS synthesis (English)..."
curl -X POST "$SERVICE_URL/synthesize" \
  -F "text=Hello, I am assisting you with IndiGo Airlines. What is your PNR?" \
  -F "language=en" \
  -o output_english.wav

if [ -f output_english.wav ]; then
    SIZE=$(ls -lh output_english.wav | awk '{print $5}')
    echo "   ✅ Generated output_english.wav ($SIZE)"
else
    echo "   ❌ Failed to generate audio"
fi
echo ""

echo "✅ Testing complete!"
echo ""
echo "Generated files:"
ls -lh output_*.wav 2>/dev/null || echo "No output files found"

