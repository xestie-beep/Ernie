#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .

echo
echo "Linux bootstrap complete."
echo "Next useful checks:"
echo "  source .venv/bin/activate"
echo "  python -m memory_agent.cli model-status"
echo "  python -m memory_agent.cli pilot-chat --no-model"
