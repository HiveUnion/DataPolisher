#!/usr/bin/env bash
# Build DataPolisher.app for Apple Silicon using Homebrew Python.
#
# Apple’s Xcode-bundled Python 3.9 links against a Tk that aborts on recent
# macOS (e.g. 26.x) with:
#   macOS 26 (2602) or later required, have instead 16 (1602) !
# Homebrew python@3.12 ships a working Tcl/Tk — use it for GUI + PyInstaller.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_PY="/opt/homebrew/opt/python@3.12/bin/python3.12"
PY="${BUILD_PYTHON:-$DEFAULT_PY}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected Python at: $PY"
  echo "Install: brew install python@3.12"
  exit 1
fi

if ! "$PY" -c "import tkinter as _tk; r=_tk.Tk(); r.destroy()" 2>/dev/null; then
  echo "ERROR: Tkinter does not start with $PY — install python-tk via Homebrew."
  exit 1
fi

VENV="${ROOT}/.venv-arm64"
if [[ ! -d "$VENV" ]] || [[ "${REBUILD_VENV:-}" == "1" ]]; then
  rm -rf "$VENV"
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip wheel
python -m pip install -q "pyinstaller>=6.0" \
  pillow opencv-python-headless "numpy>=1.26" \
  "paddlepaddle>=3.0" "paddleocr>=3.0" appdirs

echo "Running build_app.py with $(python -V) …"
python build_app.py

DEST="${1:-${DESKTOP:-$HOME/Desktop}/DataPolisher-arm64.app}"
rm -rf "$DEST"
cp -R "${ROOT}/dist/DataPolisher.app" "$DEST"
echo "Copied to: $DEST"
