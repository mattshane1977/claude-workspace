#!/usr/bin/env bash
# Start the AI Trader web server (which manages the trader subprocess via the UI).
# Dashboard: http://10.10.15.75:8080

set -e
cd "$(dirname "$0")"

mkdir -p logs data

# Use venv if present, else system python
if [ -f venv/bin/python ]; then
  PYTHON=venv/bin/python
  PIP=venv/bin/pip
  UV=venv/bin/uvicorn
else
  PYTHON=python3
  PIP=pip3
  UV=uvicorn
fi

echo "Checking dependencies..."
$PIP install -q -r requirements.txt

echo ""
echo "  AI Trader starting"
echo "  Dashboard -> http://10.10.15.75:8080"
echo ""

exec $UV web.app:app --host 0.0.0.0 --port 8080
