"""Cross-platform PyInstaller wrapper.

Run `python build_app.py` on macOS or Windows after installing
`requirements-dev.txt`. Produces a one-folder bundle in `dist/`:

* macOS  -> `dist/DataPolisher.app`
* Windows -> `dist/DataPolisher/DataPolisher.exe`

## macOS OCR

Install PyObjC Vision **before building** if you want the smaller native OCR path
at runtime (optional):

    pip install pyobjc-framework-Vision pyobjc-framework-Quartz

The app bundle **always** includes PaddlePaddle/PaddleOCR so OCR still works when
Vision is unavailable inside the frozen executable.

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
    # NOTE: Do NOT exclude pandas / google.protobuf — paddleocr>=3 + paddlex
    # import them at startup; excluding breaks ``import paddleocr`` in bundles.
    # Do NOT exclude pkg_resources._vendor — breaks setuptools/appdirs in frozen apps.
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
    "lxml",
    "sqlalchemy",
    "aiohttp",
    "grpc",
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


def _pyinstaller_copy_metadata_flags() -> list[str]:
    """Flags so PaddleX extras checks work inside a PyInstaller bundle.

    PaddleOCR 3.x builds PaddleX pipelines using ``importlib.metadata`` (versions,
    extras). Frozen apps omit ``*.dist-info`` unless we copy it — runtime then
    raises ``RuntimeError: A dependency error occurred during pipeline creation``
    even though the code is bundled. Official workaround:
    https://www.paddleocr.ai/main/version3.x/deployment/packaging.html
    """
    try:
        import importlib.metadata

        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name

        from paddlex.utils import deps as pdx_deps
    except Exception:
        return []

    needed: set[str] = set()
    for key in pdx_deps.BASE_DEP_SPECS:
        needed.add(canonicalize_name(key))
    # OCR pipelines validate the ``ocr`` / ``ocr-core`` extras graph.
    for extra in ("ocr", "ocr-core"):
        block = pdx_deps.EXTRAS.get(extra) or {}
        for dep_specs in block.values():
            for dep_spec in dep_specs:
                needed.add(canonicalize_name(Requirement(dep_spec).name))
    for explicit in ("paddlex", "paddleocr", "paddlepaddle"):
        needed.add(canonicalize_name(explicit))

    flags: list[str] = []
    seen: set[str] = set()
    for dist in importlib.metadata.distributions():
        raw_name = dist.metadata.get("Name")
        if not raw_name:
            continue
        if canonicalize_name(raw_name) not in needed:
            continue
        if raw_name in seen:
            continue
        seen.add(raw_name)
        flags += ["--copy-metadata", raw_name]

    return flags


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

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name=DataPolisher",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=appdirs",
        "--collect-submodules=data_polisher",
        # Always ship Paddle — Vision may fail inside a frozen .app on some machines,
        # and slim Vision-only bundles surface misleading “PaddleOCR missing” errors.
        "--collect-all=paddleocr",
        "--collect-all=paddle",
        "--collect-data=paddlex",
        "--collect-submodules=paddle",
    ]

    meta_flags = _pyinstaller_copy_metadata_flags()
    if meta_flags:
        cmd += meta_flags
        names = meta_flags[1::2]
        print("PyInstaller --copy-metadata for PaddleX OCR checks:", ", ".join(names))

    if platform.system() == "Darwin" and _apple_vision_available():
        print("Apple Vision PyObjC detected — also bundling Paddle for OCR fallback.")
        cmd += [
            "--hidden-import=objc",
            "--hidden-import=Vision",
            "--hidden-import=Quartz",
            "--hidden-import=Foundation",
            "--collect-submodules=objc",
            "--collect-submodules=Vision",
            "--collect-submodules=Quartz",
            "--collect-submodules=Foundation",
        ]
    elif platform.system() == "Darwin":
        print("Apple Vision PyObjC not in build env — bundling PaddleOCR only.")

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
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        return rc

    if platform.system() == "Darwin":
        _fix_macos_zlib_conflict(DIST / "DataPolisher.app")

    return 0


def _fix_macos_zlib_conflict(app_bundle: Path) -> None:
    """Replace bundled zlib-ng with system zlib to avoid symbol conflicts.

    Pillow wheels ship ``libz.1.3.1.zlib-ng.dylib`` (zlib-ng), while
    Python's own ``zlib`` C extension links against ``/usr/lib/libz.1.dylib``
    (standard zlib).  When both are loaded in the same process they export
    identical C symbols (inflate, deflate, …) and can corrupt each other's
    internal stream state, causing ``zlib.error: Error -3 … incorrect header
    check`` at runtime.

    Fix: rewrite every ``@rpath/libz.1.3.1.zlib-ng.dylib`` reference inside
    the bundle to point at the always-available system library, then delete
    the now-unused zlib-ng files.
    """
    import os

    SYSTEM_LIBZ = "/usr/lib/libz.1.dylib"
    ZLIB_NG_NAME = "libz.1.3.1.zlib-ng.dylib"

    frameworks = app_bundle / "Contents" / "Frameworks"
    if not frameworks.exists():
        return

    # Collect all Mach-O binaries (.so / .dylib) in the bundle.
    binaries: list[Path] = []
    for root, _dirs, files in os.walk(frameworks):
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix in (".so", ".dylib") and not fpath.is_symlink():
                binaries.append(fpath)

    # Patch each binary that references the bundled zlib-ng.
    patched: list[Path] = []
    for binary in binaries:
        result = subprocess.run(
            ["otool", "-L", str(binary)],
            capture_output=True, text=True,
        )
        if ZLIB_NG_NAME not in result.stdout:
            continue
        # Find the exact install-name string used (may start with @rpath or
        # an absolute path, depending on how PyInstaller laid things out).
        old_name = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if ZLIB_NG_NAME in line:
                old_name = line.split("(")[0].strip()
                break
        if not old_name:
            continue
        rc = subprocess.call([
            "install_name_tool", "-change", old_name, SYSTEM_LIBZ, str(binary),
        ])
        if rc == 0:
            subprocess.call(["codesign", "--sign", "-", "--force", str(binary)])
            patched.append(binary)
            print(f"  patched {binary.name}: {old_name} -> {SYSTEM_LIBZ}")

    if patched:
        # Remove all bundled zlib-ng copies (real files and symlinks, including
        # dangling symlinks whose target was already deleted).
        for search_root in (frameworks, app_bundle / "Contents" / "Resources"):
            if not search_root.exists():
                continue
            for root, _dirs, files in os.walk(search_root):
                for fname in files:
                    if fname == ZLIB_NG_NAME:
                        fpath = Path(root) / fname
                        fpath.unlink(missing_ok=True)
                        print(f"  removed {fpath.relative_to(app_bundle)}")
        print(f"zlib-ng conflict fix: patched {len(patched)} binaries, using {SYSTEM_LIBZ}")
    else:
        print("zlib-ng conflict fix: nothing to patch")


if __name__ == "__main__":
    raise SystemExit(main())
