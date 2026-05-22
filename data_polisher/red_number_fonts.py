"""Bundled RED Number digit fonts (metrics style aligned with 小红书 data UI).

Fonts ship under ``static/fonts/``. PIL loads files by path; native UI
consumers need OS-level registration on macOS / Windows so
``family="RED Number"`` resolves.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from functools import lru_cache
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_STATIC_FONTS = _PKG_DIR / "static" / "fonts"

RED_NUMBER_BOLD = _STATIC_FONTS / "REDNumber-Bold.otf"
RED_NUMBER_MEDIUM = _STATIC_FONTS / "REDNumber-Medium.ttf"
RED_NUMBER_REGULAR = _STATIC_FONTS / "REDNumber-Regular.ttf"

# macOS / Windows after registration — matches name ID 1 / typographic family.
RED_NUMBER_FAMILY = "RED Number"


def bundled_red_number_paths_exist() -> bool:
    return RED_NUMBER_BOLD.is_file()


def path_for_pil_weight(weight: str) -> Path | None:
    """Map template weights to bundled files."""

    w = (weight or "medium").lower()
    if w == "bold":
        p = RED_NUMBER_BOLD
    elif w == "medium":
        p = RED_NUMBER_MEDIUM
    else:
        p = RED_NUMBER_REGULAR
    return p if p.is_file() else None


def bundled_font_paths_in_order() -> list[str]:
    """Existing CLI logic expects path strings; prefer Bold → Medium → Regular."""

    out: list[str] = []
    for p in (RED_NUMBER_BOLD, RED_NUMBER_MEDIUM, RED_NUMBER_REGULAR):
        if p.is_file():
            out.append(str(p))
    return out


def prefer_red_number_metric_render() -> bool:
    """When bundled RED Number is present, draw metric digits with it (not screenshot glyphs)."""

    return RED_NUMBER_BOLD.is_file()


@lru_cache(maxsize=1)
def register_red_number_for_gui() -> bool:
    """Register bundled fonts with the OS for the current process."""

    paths = [p for p in (RED_NUMBER_REGULAR, RED_NUMBER_MEDIUM, RED_NUMBER_BOLD) if p.is_file()]
    if not paths:
        return False
    if sys.platform == "darwin":
        return any(_mac_register_font(p) for p in paths)
    if sys.platform == "win32":
        return any(_win_register_font(p) for p in paths)
    # Linux: PIL loads via file path; UI font discovery depends on the user's fontconfig setup.
    return False


def _mac_register_font(path: Path) -> bool:
    try:
        path = path.resolve()
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        ct = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreText"))
    except Exception:
        return False

    CFURLCreateFromFileSystemRepresentation = cf.CFURLCreateFromFileSystemRepresentation
    CFURLCreateFromFileSystemRepresentation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.c_bool,
    ]
    CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p

    CFRelease = cf.CFRelease
    CFRelease.argtypes = [ctypes.c_void_p]

    raw = os.fsencode(path)
    buf = ctypes.create_string_buffer(raw + b"\0")
    url = CFURLCreateFromFileSystemRepresentation(None, buf, len(raw), False)
    if not url:
        return False

    CTFontManagerRegisterFontsForURL = ct.CTFontManagerRegisterFontsForURL
    CTFontManagerRegisterFontsForURL.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool

    err = ctypes.c_void_p()
    scope = 1  # kCTFontManagerScopeProcess
    ok = CTFontManagerRegisterFontsForURL(url, scope, ctypes.byref(err))
    CFRelease(url)
    if err:
        CFRelease(err)
    return bool(ok)


def _win_register_font(path: Path) -> bool:
    try:
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        add = gdi32.AddFontResourceExW
        add.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_void_p]
        add.restype = ctypes.c_int
        FR_PRIVATE = 0x10
        n = add(str(path.resolve()), FR_PRIVATE, None)
        return n > 0
    except Exception:
        return False
