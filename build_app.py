"""Cross-platform PyInstaller wrapper.

Run `python build_app.py` on macOS or Windows after installing
`requirements-dev.txt`. Produces a one-folder bundle in `dist/`:

* macOS  -> `dist/DataPolisher.app`
* Windows -> `dist/DataPolisher/DataPolisher.exe`

We deliberately use the `--collect-all paddleocr` and `--collect-all paddle`
options because PaddleOCR ships large data files and dynamic loaders that
PyInstaller does not detect automatically.
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


def main() -> int:
    if shutil.which("pyinstaller") is None:
        print("PyInstaller not installed. Run: pip install -r requirements-dev.txt", file=sys.stderr)
        return 1

    for path in (DIST, BUILD, SPEC):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--name=DataPolisher",
        "--collect-all=paddleocr",
        "--collect-all=paddle",
        "--collect-data=paddlex",
        "--collect-submodules=paddle",
        "--hidden-import=PIL._tkinter_finder",
    ]

    if platform.system() == "Darwin":
        cmd.append("--windowed")
    elif platform.system() == "Windows":
        cmd.append("--noconsole")

    cmd.append(str(ROOT / "data_polisher" / "gui.py"))

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
