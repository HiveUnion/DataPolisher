"""Compatibility entry point for the DataPolisher desktop UI."""

from __future__ import annotations

from .web_gui import _parse_int_range_text, _parse_range_pair, main

__all__ = ["_parse_int_range_text", "_parse_range_pair", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
