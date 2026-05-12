"""macOS Apple Vision OCR backend.

Uses the OS-native Vision.framework (via PyObjC) so no model files or
PaddlePaddle/PaddleOCR packages are needed in the bundle on macOS.

Required packages (macOS only):
    pip install pyobjc-framework-Vision pyobjc-framework-Quartz
"""

from __future__ import annotations

import io
import sys
from typing import Dict, List, Optional


def is_available() -> bool:
    """Return True when running on macOS with PyObjC Vision bindings installed."""
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
        return True
    except ImportError:
        return False


def _to_pil(image):
    """Accept a PIL Image or a numpy ndarray and return a PIL Image."""
    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        return image
    try:
        import numpy as np  # type: ignore

        if isinstance(image, np.ndarray):
            return PILImage.fromarray(image)
    except ImportError:
        pass
    raise TypeError(f"Unsupported image type: {type(image)}")


def _pil_to_cg_image(pil_image):
    """Convert a PIL Image to a CGImageRef for use with Apple Vision."""
    import Quartz
    from Foundation import NSData

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    raw = buf.getvalue()
    ns_data = NSData.dataWithBytes_length_(raw, len(raw))
    source = Quartz.CGImageSourceCreateWithData(ns_data, None)
    return Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)


def detect_items(image) -> List[Dict]:
    """Return a list of ``{text, rect}`` dicts using Apple Vision OCR.

    ``rect`` uses pixel coordinates with the top-left origin (same convention
    as PaddleOCR helpers in this package).
    """
    import Vision

    pil = _to_pil(image).convert("RGB")
    w, h = pil.size
    cg = _pil_to_cg_image(pil)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLanguages_(["zh-Hans", "zh-Hant", "en-US"])
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    success, _error = handler.performRequests_error_([request], None)
    if not success:
        return []

    items = []
    for obs in request.results() or []:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string())

        bb = obs.boundingBox()
        # Vision uses normalized coordinates with the origin at *bottom-left*.
        # Convert to pixel coordinates with the origin at top-left.
        px = bb.origin.x * w
        py = (1.0 - bb.origin.y - bb.size.height) * h
        pw = bb.size.width * w
        ph = bb.size.height * h

        items.append(
            {
                "text": text,
                "rect": {
                    "x": int(px),
                    "y": int(py),
                    "width": max(1, int(pw)),
                    "height": max(1, int(ph)),
                },
            }
        )

    return items


def detect_bounds(image) -> Optional[Dict]:
    """Return the bounding box that covers all numeric text regions, or None."""
    items = detect_items(image)
    min_x = min_y = 10**9
    max_x = max_y = -1
    matched = False

    for item in items:
        stripped = "".join(ch for ch in str(item["text"]) if ch.isdigit() or ch == ".")
        if not stripped:
            continue
        r = item["rect"]
        min_x = min(min_x, r["x"])
        min_y = min(min_y, r["y"])
        max_x = max(max_x, r["x"] + r["width"])
        max_y = max(max_y, r["y"] + r["height"])
        matched = True

    if not matched or max_x <= min_x or max_y <= min_y:
        return None

    return {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y}
