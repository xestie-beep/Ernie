#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv/bin/python3" ]]; then
  echo "Missing virtualenv at .venv."
  echo "Run ./scripts/bootstrap_linux.sh first."
  exit 1
fi

CONFIG_DIR="${HOME}/.config/ernie-cockpit"
ENV_PATH="${CONFIG_DIR}/remote.env"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_PATH="${UNIT_DIR}/ernie-cockpit-remote.service"
PORT="8766"
DISPLAY_HOST=""
TOKEN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:?missing port value}"
      shift 2
      ;;
    --display-host)
      DISPLAY_HOST="${2:?missing display host value}"
      shift 2
      ;;
    --token)
      TOKEN="${2:?missing token value}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

mkdir -p "$CONFIG_DIR" "$UNIT_DIR"

if [[ -z "$DISPLAY_HOST" ]] && command -v tailscale >/dev/null 2>&1; then
  DISPLAY_HOST="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$TOKEN" ]]; then
  if [[ -f "$ENV_PATH" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_PATH"
    TOKEN="${TOKEN:-}"
  fi
fi

if [[ -z "$TOKEN" ]]; then
  TOKEN="$(uuidgen | tr '[:upper:]' '[:lower:]')"
fi

cat >"$ENV_PATH" <<EOF
PORT="$PORT"
TOKEN="$TOKEN"
DISPLAY_HOST="$DISPLAY_HOST"
EOF

cat >"$UNIT_PATH" <<EOF
[Unit]
Description=Ernie Cockpit remote service
After=default.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
EnvironmentFile=$ENV_PATH
ExecStart=$ROOT_DIR/.venv/bin/python3 -m memory_agent.cli serve --host 0.0.0.0 --port \${PORT} --token \${TOKEN} --display-host \${DISPLAY_HOST}
Restart=on-failure
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now ernie-cockpit-remote.service

echo "Installed remote user service."
echo "  unit: $UNIT_PATH"
echo "  config: $ENV_PATH"
if [[ -n "$DISPLAY_HOST" ]]; then
  echo "  url: http://${DISPLAY_HOST}:${PORT}/"
fi
echo "  token: $TOKEN"
echo "  status: systemctl --user status ernie-cockpit-remote.service"
