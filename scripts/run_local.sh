#!/bin/bash
# Quick script to run TTS service locally without model

set -e

cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment to disable model loading
export MODEL_LOADING_ENABLED=false
export REQUIRE_MODEL=false

echo "🚀 Starting TTS service in local testing mode (no model required)..."
echo "   - Model loading: DISABLED"
echo "   - Endpoints will return 503 for synthesis (expected)"
echo "   - Health/readiness endpoints will work"
echo ""

uvicorn app:app --reload --host 0.0.0.0 --port 8000

