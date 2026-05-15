from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Pixel = Tuple[int, int, int]
PixelGrid = Sequence[Sequence[Pixel]]


def format_percent(value: float) -> str:
    percent = round(value * 100, 1)
    if percent == int(percent):
        return f"{int(percent)}%"
    return f"{percent:.1f}%"


def calculate_metrics(
    exposure: float,
    views: float,
    likes: float,
    comments: float,
    collects: float,
    shares: float,
) -> Dict[str, object]:
    exposure = float(exposure)
    views = float(views)
    parts = [float(likes), float(comments), float(collects), float(shares)]

    if exposure <= 0:
        raise ValueError("exposure must be greater than 0")
    if views < 0:
        raise ValueError("views must not be negative")
    if any(value < 0 for value in parts):
        raise ValueError("interaction values must not be negative")

    interaction_count = sum(parts)
    click_rate = views / exposure
    interaction_rate = interaction_count / views if views > 0 else 0
    views_txt = str(round(views))

    return {
        "exposure": round(exposure),
        "views": round(views),
        "interaction_count": round(interaction_count),
        "click_rate": click_rate,
        "interaction_rate": interaction_rate,
        "exposure_text": str(round(exposure)),
        "views_text": views_txt,
        "header_views_text": views_txt,
        "click_rate_text": format_percent(click_rate),
        "interaction_rate_text": format_percent(interaction_rate),
    }


def pixel_looks_like_text(pixel: Pixel) -> bool:
    r, g, b = pixel[:3]
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 185 and max_channel - min_channel < 80


def detect_dark_text_bounds(pixels: PixelGrid) -> Optional[Dict[str, int]]:
    min_x = 10**9
    min_y = 10**9
    max_x = -1
    max_y = -1
    count = 0

    for y, row in enumerate(pixels):
        for x, pixel in enumerate(row):
            if pixel_looks_like_text(pixel):
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                count += 1

    if count < 4 or max_x < min_x or max_y < min_y:
        return None

    return {
        "x": min_x,
        "y": min_y,
        "width": max_x - min_x + 1,
        "height": max_y - min_y + 1,
    }


def expand_rect(rect: Dict[str, int], padding: int, limit: Dict[str, int]) -> Dict[str, int]:
    x = max(0, int(rect["x"]) - padding)
    y = max(0, int(rect["y"]) - padding)
    right = min(int(limit["width"]), int(rect["x"] + rect["width"]) + padding)
    bottom = min(int(limit["height"]), int(rect["y"] + rect["height"]) + padding)
    return {
        "x": x,
        "y": y,
        "width": max(1, right - x),
        "height": max(1, bottom - y),
    }


def scale_rect(rect: Dict[str, int], source_size: Dict[str, int], target_size: Dict[str, int]) -> Dict[str, int]:
    scale_x = target_size["width"] / source_size["width"]
    scale_y = target_size["height"] / source_size["height"]
    return {
        "x": round(rect["x"] * scale_x),
        "y": round(rect["y"] * scale_y),
        "width": round(rect["width"] * scale_x),
        "height": round(rect["height"] * scale_y),
    }


def rect_to_box(rect: Dict[str, int]) -> Tuple[int, int, int, int]:
    return (
        int(rect["x"]),
        int(rect["y"]),
        int(rect["x"] + rect["width"]),
        int(rect["y"] + rect["height"]),
    )

