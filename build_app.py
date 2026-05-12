"""Cross-platform PyInstaller wrapper.

Run `python build_app.py` on macOS or Windows after installing
`requirements-dev.txt`. Produces a one-folder bundle in `dist/`:

* macOS  -> `dist/DataPolisher.app`
* Windows -> `dist/DataPolisher/DataPolisher.exe`

## macOS size optimisation (recommended)

Install the Apple Vision backend before building:

    pip install pyobjc-framework-Vision pyobjc-framework-Quartz

When those packages are present this script automatically skips bundling
PaddlePaddle / PaddleOCR and uses the OS-native Vision.framework instead,
reducing the app bundle by ~300–600 MB.

## Windows / fallback

On Windows (or macOS without PyObjC) the script falls back to bundling the
full PaddleOCR stack as before.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = ROOT / "DataPolisher.spec"

# ---------------------------------------------------------------------------
# Modules that are never needed at runtime but are often dragged in
# transitively by large packages.  Excluding them shaves a meaningful amount
# off the bundle without any risk of breaking the app.
# ---------------------------------------------------------------------------
_EXCLUDE_MODULES = [
    "matplotlib",
    "matplotlib.backends",
    "scipy",
    "IPython",
    "ipykernel",
    "ipywidgets",
    "jupyter",
    "notebook",
    "nbformat",
    "nbconvert",
    "pandas",
    "sklearn",
    "skimage",
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "numba",
    "llvmlite",
    "zmq",
    "tornado",
    "cryptography",
    "Crypto",
    "docutils",
    "sphinx",
    "pytest",
    "pkg_resources._vendor",
    "lxml",
    "sqlalchemy",
    "aiohttp",
    "grpc",
    "google.protobuf",
]


def _apple_vision_available() -> bool:
    """Return True when the Apple Vision OCR backend is usable right now."""
    if platform.system() != "Darwin":
        return False
    try:
        import Vision  # type: ignore  # noqa: F401
        import Quartz  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not installed. Run: pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    for path in (DIST, BUILD, SPEC):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    use_apple_vision = _apple_vision_available()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name=DataPolisher",
        "--hidden-import=PIL._tkinter_finder",
        "--collect-submodules=data_polisher",
    ]

    if use_apple_vision:
        print("Apple Vision OCR detected -- skipping PaddlePaddle/PaddleOCR bundle.")
        # PyObjC frameworks are loaded dynamically; tell PyInstaller about them.
        cmd += [
            "--hidden-import=objc",
            "--hidden-import=Vision",
            "--hidden-import=Quartz",
            "--hidden-import=Foundation",
            "--collect-submodules=objc",
        ]
    else:
        print("Apple Vision not available -- bundling PaddleOCR (large).")
        print("  To shrink the bundle: pip install pyobjc-framework-Vision pyobjc-framework-Quartz")
        cmd += [
            "--collect-all=paddleocr",
            "--collect-all=paddle",
            "--collect-data=paddlex",
            "--collect-submodules=paddle",
        ]

    # Exclude heavy modules that are never needed.
    for mod in _EXCLUDE_MODULES:
        cmd += ["--exclude-module", mod]

    if platform.system() == "Darwin":
        cmd.append("--windowed")
        # Strip debug symbols from all collected binaries — saves ~20-30 %.
        cmd.append("--strip")
    elif platform.system() == "Windows":
        cmd.append("--noconsole")

    # Use a top-level launcher so relative imports inside the package work.
    cmd.append(str(ROOT / "launcher.py"))

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
