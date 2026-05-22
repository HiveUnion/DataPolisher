# Shared macOS PyInstaller steps — sourced by build_mac_{arm64,intel}_app.sh
# Requires exported before source:
#   BUILD_MAC_PY_DEFAULT  — full path to Homebrew python3.12
#   BUILD_MAC_VENV_TAG    — short suffix for .venv-<tag>
#   BUILD_MAC_APP_TAG     — output bundle name fragment DataPolisher-<tag>.app

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${BUILD_MAC_PY_DEFAULT:?BUILD_MAC_PY_DEFAULT must be set by wrapper}"
: "${BUILD_MAC_VENV_TAG:?BUILD_MAC_VENV_TAG must be set by wrapper}"
: "${BUILD_MAC_APP_TAG:?BUILD_MAC_APP_TAG must be set by wrapper}"

PY="${BUILD_PYTHON:-$BUILD_MAC_PY_DEFAULT}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected Python at: $PY"
  echo "Install: brew install python@3.12"
  exit 1
fi

VENV="${ROOT}/.venv-${BUILD_MAC_VENV_TAG}"
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
# GUI 依赖 pywebview（见 pyproject.toml）；安装 editable 包时一并拉取。
python -m pip install -q -e "$ROOT"

echo "Running build_app.py with $(python -V) ($(uname -m)) …"
python build_app.py

DEST="${1:-${DESKTOP:-$HOME/Desktop}/DataPolisher-${BUILD_MAC_APP_TAG}.app}"
rm -rf "$DEST"
cp -R "${ROOT}/dist/DataPolisher.app" "$DEST"
echo "Copied to: $DEST"
