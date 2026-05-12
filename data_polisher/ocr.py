from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional


def _apple_vision_available() -> bool:
    """Return True when the Apple Vision OCR backend can be used."""
    if sys.platform != "darwin":
        return False
    try:
        from . import ocr_apple  # noqa: F401

        return ocr_apple.is_available()
    except Exception:
        return False


def _flatten_ocr_items(result: Any) -> Iterable[Any]:
    if not result:
        return []
    if isinstance(result, list):
        if result and isinstance(result[0], list) and result[0] and isinstance(result[0][0], list):
            items = []
            for page in result:
                items.extend(page)
            return items
        return result
    return []


def _is_metric_text(text: str) -> bool:
    stripped = "".join(ch for ch in str(text) if ch.isdigit() or ch == ".")
    return bool(stripped)


def _update_bounds(bounds: Dict[str, int], x1: int, y1: int, x2: int, y2: int) -> None:
    bounds["min_x"] = min(bounds["min_x"], x1)
    bounds["min_y"] = min(bounds["min_y"], y1)
    bounds["max_x"] = max(bounds["max_x"], x2)
    bounds["max_y"] = max(bounds["max_y"], y2)


def extract_ocr_text_bounds(result: Any) -> Optional[Dict[str, int]]:
    bounds = {"min_x": 10**9, "min_y": 10**9, "max_x": -1, "max_y": -1}
    matched = False

    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if isinstance(page, dict):
            texts = page.get("rec_texts")
            boxes = page.get("rec_boxes")
            if texts is None or boxes is None:
                continue
            for text, box in zip(texts, boxes):
                if not _is_metric_text(str(text)):
                    continue
                values = list(box)
                if len(values) >= 4:
                    _update_bounds(bounds, int(values[0]), int(values[1]), int(values[2]), int(values[3]))
                    matched = True

    for item in _flatten_ocr_items(result):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box = item[0]
        payload = item[1]
        text = payload[0] if isinstance(payload, (list, tuple)) and payload else ""
        if not _is_metric_text(str(text)):
            continue
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue

        xs = [point[0] for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
        ys = [point[1] for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not xs or not ys:
            continue

        _update_bounds(bounds, int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        matched = True

    if not matched or bounds["max_x"] <= bounds["min_x"] or bounds["max_y"] <= bounds["min_y"]:
        return None

    return {
        "x": bounds["min_x"],
        "y": bounds["min_y"],
        "width": bounds["max_x"] - bounds["min_x"],
        "height": bounds["max_y"] - bounds["min_y"],
    }


def extract_ocr_items(result: Any):
    items = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if isinstance(page, dict):
            texts = page.get("rec_texts")
            boxes = page.get("rec_boxes")
            if texts is None or boxes is None:
                continue
            for text, box in zip(texts, boxes):
                values = list(box)
                if len(values) >= 4:
                    items.append(
                        {
                            "text": str(text),
                            "rect": {
                                "x": int(values[0]),
                                "y": int(values[1]),
                                "width": int(values[2]) - int(values[0]),
                                "height": int(values[3]) - int(values[1]),
                            },
                        }
                    )
            continue

        for item in _flatten_ocr_items(page):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            box = item[0]
            payload = item[1]
            text = payload[0] if isinstance(payload, (list, tuple)) and payload else ""
            xs = [point[0] for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
            ys = [point[1] for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
            if xs and ys:
                items.append(
                    {
                        "text": str(text),
                        "rect": {
                            "x": int(min(xs)),
                            "y": int(min(ys)),
                            "width": int(max(xs) - min(xs)),
                            "height": int(max(ys) - min(ys)),
                        },
                    }
                )
    return items


@lru_cache(maxsize=1)
def get_ocr_engine():
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install OCR dependencies with "
            "`pip install -r requirements.txt` or use the bundled installer."
        ) from exc

    return PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def detect_bounds_with_paddle(image) -> Optional[Dict[str, int]]:
    if _apple_vision_available():
        from . import ocr_apple

        return ocr_apple.detect_bounds(image)
    engine = get_ocr_engine()
    result = engine.predict(image)
    return extract_ocr_text_bounds(result)


def detect_items_with_paddle(image):
    if _apple_vision_available():
        from . import ocr_apple

        return ocr_apple.detect_items(image)
    engine = get_ocr_engine()
    result = engine.predict(image)
    return extract_ocr_items(result)

