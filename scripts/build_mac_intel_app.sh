#!/usr/bin/env bash
# Build DataPolisher.app for Intel Mac (x86_64) using Homebrew Python under /usr/local.
#
# Run this on an Intel Mac (or from a Rosetta shell with x86_64 Homebrew + python@3.12).
# Apple Silicon 默认请用 build_mac_arm64_app.sh。
#
# Apple’s Xcode-bundled Python links against a Tk that may abort on recent macOS;
# Homebrew python@3.12 ships a working Tcl/Tk — use it for GUI + PyInstaller.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUILD_MAC_PY_DEFAULT="/usr/local/opt/python@3.12/bin/python3.12"
export BUILD_MAC_VENV_TAG="intel"
export BUILD_MAC_APP_TAG="intel"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_build_mac_app_body.sh"
