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
echo "  ./scripts/run_cockpit.sh start local"
echo "  ./scripts/run_cockpit.sh start remote"
echo "  ./scripts/install_desktop_launcher.sh"
echo "  ./scripts/install_user_service.sh"
echo "  ./scripts/install_remote_service.sh"
echo "  ./scripts/manage_remote_access.sh show"
echo "  ./scripts/run_cockpit.sh status"
echo "  python3 -m memory_agent.cli model-status"
