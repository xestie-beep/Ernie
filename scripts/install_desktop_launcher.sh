#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing virtualenv at .venv."
  echo "Run ./scripts/bootstrap_linux.sh first."
  exit 1
fi

BIN_DIR="${HOME}/.local/bin"
APP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/scalable/apps"
ICON_SOURCE="${ROOT_DIR}/assets/ernie-cockpit.svg"
ICON_PATH="${ICON_DIR}/ernie-cockpit.svg"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}"
LAUNCHER_PATH="${BIN_DIR}/ernie-cockpit"
DESKTOP_PATH="${APP_DIR}/ernie-cockpit.desktop"

mkdir -p "$BIN_DIR" "$APP_DIR" "$ICON_DIR" "$STATE_DIR"

cat >"$LAUNCHER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$ROOT_DIR"
cd "\$ROOT_DIR"

if command -v systemctl >/dev/null 2>&1 && systemctl --user cat ernie-cockpit.service >/dev/null 2>&1; then
  systemctl --user start ernie-cockpit.service
else
  "\$ROOT_DIR/scripts/run_cockpit.sh" start local "\$@"
fi

if command -v xdg-open >/dev/null 2>&1; then
  sleep 1
  xdg-open "http://127.0.0.1:8765/" >/dev/null 2>&1 &
fi
EOF

chmod +x "$LAUNCHER_PATH"

if [[ -f "$ICON_SOURCE" ]]; then
  cp "$ICON_SOURCE" "$ICON_PATH"
fi

cat >"$DESKTOP_PATH" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Ernie Cockpit
Comment=Launch the Ernie local cockpit
Exec=$LAUNCHER_PATH
Icon=ernie-cockpit
Terminal=false
Categories=Development;Utility;
Keywords=Ernie;Agent;Cockpit;Memory;Local AI;
StartupNotify=true
EOF

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APP_DIR" >/dev/null 2>&1 || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true
fi

echo "Installed desktop launcher."
echo "  launcher: $LAUNCHER_PATH"
echo "  desktop entry: $DESKTOP_PATH"
if [[ -f "$ICON_PATH" ]]; then
  echo "  icon: $ICON_PATH"
fi
echo "You can now launch 'Ernie Cockpit' from the desktop app menu or run:"
echo "  ernie-cockpit"
