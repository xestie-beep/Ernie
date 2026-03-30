#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${HOME}/.config/ernie-cockpit"
ENV_PATH="${CONFIG_DIR}/remote.env"
UNIT_NAME="ernie-cockpit-remote.service"

if [[ ! -f "$ENV_PATH" ]]; then
  echo "Missing remote access config at $ENV_PATH"
  echo "Run ./scripts/install_remote_service.sh first."
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_PATH"

ACTION="${1:-show}"

show_access() {
  echo "Remote cockpit access:"
  if [[ -n "${DISPLAY_HOST:-}" ]]; then
    echo "  url: http://${DISPLAY_HOST}:${PORT}/"
  else
    echo "  host: set DISPLAY_HOST in $ENV_PATH to print a browser URL"
  fi
  echo "  port: ${PORT:-8766}"
  echo "  token: ${TOKEN:-}"
}

rotate_token() {
  NEW_TOKEN="$(uuidgen | tr '[:upper:]' '[:lower:]')"
  cat >"$ENV_PATH" <<EOF
PORT="${PORT:-8766}"
TOKEN="$NEW_TOKEN"
DISPLAY_HOST="${DISPLAY_HOST:-}"
EOF
  systemctl --user restart "$UNIT_NAME"
  echo "Rotated remote access token."
  echo "  token: $NEW_TOKEN"
  if [[ -n "${DISPLAY_HOST:-}" ]]; then
    echo "  url: http://${DISPLAY_HOST}:${PORT}/"
  fi
}

case "$ACTION" in
  show)
    show_access
    ;;
  rotate)
    rotate_token
    ;;
  *)
    echo "Usage:"
    echo "  ./scripts/manage_remote_access.sh show"
    echo "  ./scripts/manage_remote_access.sh rotate"
    exit 2
    ;;
esac
