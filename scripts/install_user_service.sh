#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv/bin/python3" ]]; then
  echo "Missing virtualenv at .venv."
  echo "Run ./scripts/bootstrap_linux.sh first."
  exit 1
fi

UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_PATH="${UNIT_DIR}/ernie-cockpit.service"

mkdir -p "$UNIT_DIR"

cat >"$UNIT_PATH" <<EOF
[Unit]
Description=Ernie Cockpit local service
After=default.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
ExecStart=$ROOT_DIR/.venv/bin/python3 -m memory_agent.cli serve --host 127.0.0.1 --port 8765
Restart=on-failure
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now ernie-cockpit.service

echo "Installed user service."
echo "  unit: $UNIT_PATH"
echo "  status: systemctl --user status ernie-cockpit.service"
echo "  stop: systemctl --user stop ernie-cockpit.service"
echo "  restart: systemctl --user restart ernie-cockpit.service"
