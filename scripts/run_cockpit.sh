#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing virtualenv at .venv."
  echo "Run ./scripts/bootstrap_linux.sh first."
  exit 1
fi

source .venv/bin/activate

STATE_FILE="/tmp/ernie_cockpit_runtime.env"
DEFAULT_LOG_FILE="/tmp/ernie_cockpit.log"

runtime_is_live() {
  if [[ -z "${PID:-}" ]]; then
    return 1
  fi
  kill -0 "$PID" >/dev/null 2>&1
}

load_state() {
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  cat >"$STATE_FILE" <<EOF
PID="$PID"
MODE="$MODE"
HOST="$HOST"
PORT="$PORT"
TOKEN="$TOKEN"
DISPLAY_HOST="$DISPLAY_HOST"
ACCESS_URL="$ACCESS_URL"
LOG_FILE="$LOG_FILE"
EOF
}

print_status() {
  load_state
  if ! runtime_is_live; then
    echo "Ernie cockpit is not running under the launcher."
    if [[ -f "$STATE_FILE" ]]; then
      echo "Stale runtime file: $STATE_FILE"
    fi
    return 1
  fi
  echo "Ernie cockpit is running."
  echo "  pid: $PID"
  echo "  mode: $MODE"
  echo "  host: $HOST"
  echo "  port: $PORT"
  echo "  log: $LOG_FILE"
  if [[ -n "${ACCESS_URL:-}" ]]; then
    echo "  url: $ACCESS_URL"
  fi
  if [[ -n "${TOKEN:-}" ]]; then
    echo "  token: $TOKEN"
  fi
}

stop_runtime() {
  load_state
  if ! runtime_is_live; then
    echo "Ernie cockpit is not running under the launcher."
    rm -f "$STATE_FILE"
    return 0
  fi
  kill "$PID"
  rm -f "$STATE_FILE"
  echo "Stopped Ernie cockpit (pid $PID)."
}

resolve_display_host() {
  if [[ -n "${DISPLAY_HOST:-}" ]]; then
    return 0
  fi
  if command -v tailscale >/dev/null 2>&1; then
    DISPLAY_HOST="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
  fi
}

build_access_url() {
  ACCESS_URL=""
  if [[ "$MODE" == "local" ]]; then
    ACCESS_URL="http://127.0.0.1:${PORT}/"
    return
  fi
  if [[ -n "${DISPLAY_HOST:-}" ]]; then
    ACCESS_URL="http://${DISPLAY_HOST}:${PORT}/"
    if [[ -n "${TOKEN:-}" ]]; then
      ACCESS_URL="${ACCESS_URL}?token=${TOKEN}"
    fi
  fi
}

start_runtime() {
  MODE="${1:-local}"
  shift || true

  HOST="127.0.0.1"
  PORT="8765"
  TOKEN=""
  DISPLAY_HOST=""
  LOG_FILE="$DEFAULT_LOG_FILE"
  EXTRA_ARGS=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --host)
        HOST="${2:?missing host value}"
        shift 2
        ;;
      --port)
        PORT="${2:?missing port value}"
        shift 2
        ;;
      --token)
        TOKEN="${2:?missing token value}"
        shift 2
        ;;
      --display-host)
        DISPLAY_HOST="${2:?missing display host value}"
        shift 2
        ;;
      --log-file)
        LOG_FILE="${2:?missing log file value}"
        shift 2
        ;;
      *)
        EXTRA_ARGS+=("$1")
        shift
        ;;
    esac
  done

  if [[ "$MODE" != "local" && "$MODE" != "remote" ]]; then
    echo "Unsupported mode: $MODE"
    exit 2
  fi

  load_state
  if runtime_is_live; then
    echo "Ernie cockpit is already running under the launcher."
    print_status
    return 0
  fi

  if [[ "$MODE" == "remote" ]]; then
    HOST="0.0.0.0"
    if [[ -z "$TOKEN" ]]; then
      TOKEN="$(uuidgen | tr '[:upper:]' '[:lower:]')"
    fi
    resolve_display_host
  fi

  build_access_url

  CMD=(python3 -m memory_agent.cli serve --host "$HOST" --port "$PORT")
  if [[ -n "$TOKEN" ]]; then
    CMD+=(--token "$TOKEN")
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  if command -v setsid >/dev/null 2>&1; then
    setsid "${CMD[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
  else
    nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  fi
  PID="$!"
  sleep 1
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "Ernie cockpit failed to stay up."
    if [[ -f "$LOG_FILE" ]]; then
      echo "Last log lines:"
      tail -n 20 "$LOG_FILE"
    fi
    exit 1
  fi
  save_state

  echo "Started Ernie cockpit."
  echo "  pid: $PID"
  echo "  mode: $MODE"
  echo "  log: $LOG_FILE"
  if [[ -n "$ACCESS_URL" ]]; then
    echo "  url: $ACCESS_URL"
  fi
  if [[ -n "$TOKEN" ]]; then
    echo "  token: $TOKEN"
  fi
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_cockpit.sh start [local|remote] [options]
  ./scripts/run_cockpit.sh stop
  ./scripts/run_cockpit.sh restart [local|remote] [options]
  ./scripts/run_cockpit.sh status

Backward-compatible shortcuts:
  ./scripts/run_cockpit.sh local [options]
  ./scripts/run_cockpit.sh remote [options]

Options:
  --host HOST
  --port PORT
  --token TOKEN
  --display-host HOST_OR_IP
  --log-file PATH

Examples:
  ./scripts/run_cockpit.sh start local
  ./scripts/run_cockpit.sh start remote
  ./scripts/run_cockpit.sh start remote --token YOUR_SHARED_TOKEN --display-host 100.125.133.85
  ./scripts/run_cockpit.sh status
  ./scripts/run_cockpit.sh stop
EOF
}

ACTION="${1:-start}"
case "$ACTION" in
  start)
    shift || true
    start_runtime "${1:-local}" "${@:2}"
    ;;
  restart)
    shift || true
    stop_runtime
    start_runtime "${1:-local}" "${@:2}"
    ;;
  stop)
    stop_runtime
    ;;
  status)
    print_status
    ;;
  local|remote)
    start_runtime "$ACTION" "${@:2}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
