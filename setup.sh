#!/bin/bash
set -e

echo "=== AI Trader Setup ==="

# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Create .env if not present
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Created .env — fill in your Alpaca API keys before running."
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Edit .env with your Alpaca API key/secret"
echo "  2. Make sure Ollama is running: ollama serve"
echo "  3. Pull your model: ollama pull qwen2.5:32b"
echo "  4. source .venv/bin/activate"
echo "  5. python main.py"
