#!/usr/bin/env bash
# Veritas AI - One-Click Launcher for 8GB Laptop Profile

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo " Starting VERITAS AI: Fake News Forensics LLM"
echo " Hardware Profile: Laptop 8GB RAM | Apple Silicon / CPU"
echo "=========================================================="

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "Launching Fake News Detector on http://localhost:8501..."
python3 -m streamlit run app/app.py --server.fileWatcherType none --server.port 8501


