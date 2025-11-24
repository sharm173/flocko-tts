#!/bin/bash
# Quick setup script for local testing

set -e

echo "🚀 Setting up TTS service for local testing..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and build tools
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Try to install numpy 1.25.2 (required by chatterbox-tts)
echo "Installing numpy 1.25.2 (this may take a few minutes if building from source)..."
pip install numpy==1.25.2 || {
    echo "⚠️  Failed to install numpy 1.25.2"
    echo "Trying to install from source..."
    pip install --no-binary numpy numpy==1.25.2 || {
        echo "❌ Could not install numpy 1.25.2"
        echo "💡 Try using conda or Python 3.11 instead"
        exit 1
    }
}

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To run the service:"
echo "  source venv/bin/activate"
echo "  uvicorn app:app --reload"

