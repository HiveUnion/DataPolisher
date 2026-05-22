#!/usr/bin/env bash
# Build DataPolisher.app for Apple Silicon (arm64) using Homebrew Python.
#
# Intel Mac 请使用 scripts/build_mac_intel_app.sh（需在 x86_64 环境构建）。
#
# Homebrew python@3.12 is used for a repeatable PyInstaller build environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BUILD_MAC_PY_DEFAULT="/opt/homebrew/opt/python@3.12/bin/python3.12"
export BUILD_MAC_VENV_TAG="arm64"
export BUILD_MAC_APP_TAG="arm64"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_build_mac_app_body.sh"
