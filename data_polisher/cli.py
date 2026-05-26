from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .core import (
    calculate_metrics,
    detect_dark_text_bounds,
    expand_rect,
    pixel_looks_like_text,
    rect_to_box,
    scale_rect,
)
from .ocr import detect_bounds_with_paddle, detect_items_with_paddle
from .template import BASE_SIZE, NORMAL_FIELDS
from .red_number_fonts import (
    RED_NUMBER_BOLD,
    RED_NUMBER_MEDIUM,
    RED_NUMBER_REGULAR,
    bundled_font_paths_in_order,
    path_for_pil_weight,
    prefer_red_number_metric_render,
)

_PKG_STATIC_FONTS = Path(__file__).resolve().parent / "static" / "fonts"
# DIN ``%`` from 小红书 APK — geometric sans, closer to analytics cards than 苹方.
BUNDLED_DIN_PERCENT = _PKG_STATIC_FONTS / "DIN-OT-Medium.ttf"
# 信息流封面左下角小眼睛浏览数：使用小红书运行时下发的 FZYouHS 508R。
BUNDLED_FEED_OVERLAY_VIEWS_FONT = _PKG_STATIC_FONTS / "FZYouHS-508R.ttf"
FEED_OVERLAY_VIEWS_FONT_SIZE = 17
FEED_OVERLAY_VIEWS_DX = -2
FEED_OVERLAY_VIEWS_DY = 1
FEED_OVERLAY_VIEWS_ALPHA_GAIN = 1.0
FEED_OVERLAY_VIEWS_ALPHA_GAMMA = 0.76

_FONT_EXTRA = bundled_font_paths_in_order()

FONT_CANDIDATE_PATHS = _FONT_EXTRA + [
    "/System/Library/Fonts/KohinoorGujarati.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/SFNSRounded.ttf",
    "/System/Library/Fonts/ADTNumeric.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
]

BODY_NATIVE_FONT_PATH = (
    str(RED_NUMBER_BOLD) if RED_NUMBER_BOLD.is_file() else "/System/Library/Fonts/KohinoorGujarati.ttc"
)
BODY_NATIVE_FONT_CANDIDATE_PATHS = _FONT_EXTRA + [
    "/System/Library/Fonts/KohinoorGujarati.ttc",
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/SFNSRounded.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/ADTNumeric.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
]
BODY_NATIVE_FONT_SIZE_ADJUST = 2
BODY_NATIVE_FORCE_EDGE_VARIANT = "w1x:quantized"
BODY_NATIVE_FORCE_ALPHA_STRENGTH = 0.25
# 顶部小眼睛：字形更小、偏中等字重，不使用详情 Bold + 笔画预设。
HEADER_VIEWS_FONT_ADJUST = -2


def load_optional_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        return cv2, np
    except Exception:
        return None, None


def crop_pixels(image, rect: Dict[str, int]):
    crop = image.crop(rect_to_box(rect)).convert("RGB")
    width, height = crop.size
    raw = list(crop.getdata())
    return [raw[y * width : (y + 1) * width] for y in range(height)]


def translate_rect(rect: Dict[str, int], origin: Dict[str, int]) -> Dict[str, int]:
    return {
        "x": rect["x"] + origin["x"],
        "y": rect["y"] + origin["y"],
        "width": rect["width"],
        "height": rect["height"],
    }


def average_edge_color(image, rect: Dict[str, int]):
    import numpy as np

    width, height = image.size
    padding = max(3, min(rect["width"], rect["height"]) // 5)
    sample_rects = [
        {"x": rect["x"] - padding, "y": rect["y"], "width": padding, "height": rect["height"]},
        {"x": rect["x"] + rect["width"], "y": rect["y"], "width": padding, "height": rect["height"]},
        {"x": rect["x"], "y": rect["y"] - padding, "width": rect["width"], "height": padding},
        {"x": rect["x"], "y": rect["y"] + rect["height"], "width": rect["width"], "height": padding},
    ]

    arrays = []
    for item in sample_rects:
        clipped = {
            "x": max(0, item["x"]),
            "y": max(0, item["y"]),
            "width": max(1, min(width - max(0, item["x"]), item["width"])),
            "height": max(1, min(height - max(0, item["y"]), item["height"])),
        }
        arr = np.array(image.crop(rect_to_box(clipped)).convert("RGB"), dtype=np.float32)
        arrays.append(arr.reshape(-1, 3))

    if not arrays:
        return (255, 255, 255)

    pixels = np.concatenate(arrays, axis=0)
    luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    is_bright = (luminance > 218) & ((pixels.max(axis=1) - pixels.min(axis=1)) < 28)
    if is_bright.any():
        pixels = pixels[is_bright]

    mean = pixels.mean(axis=0)
    return (round(float(mean[0])), round(float(mean[1])), round(float(mean[2])))


def inpaint_or_fill(image, rect: Dict[str, int]):
    from PIL import ImageDraw

    patched = image.copy()
    draw = ImageDraw.Draw(patched)
    color = average_edge_color(image, rect)
    draw.rectangle(rect_to_box(rect), fill=color)
    return patched


def inpaint_overlay_views_compact_fill(image, rect: Dict[str, int]):
    """Erase overlay digits in a tight rect using local capsule-toned median (not edge slab)."""

    import numpy as np
    from PIL import ImageDraw

    patched = image.copy()
    crop = patched.crop(rect_to_box(rect))
    arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    thr = float(np.percentile(lum, 58))
    flat_lum = lum.reshape(-1)
    flat_rgb = arr.reshape(-1, 3)
    mask = flat_lum < thr
    pixels = flat_rgb[mask]
    if pixels.shape[0] < 10:
        color = average_edge_color(image, rect)
    else:
        med = np.median(pixels, axis=0)
        color = (int(round(med[0])), int(round(med[1])), int(round(med[2])))

    draw = ImageDraw.Draw(patched)
    draw.rectangle(rect_to_box(rect), fill=color)
    return patched


def inpaint_overlay_views_translucent_fill(
    image,
    source_image,
    padded: Dict[str, int],
    ink_rect: Dict[str, int],
    thumb: Dict[str, int],
    strip_roi: Dict[str, int],
):
    """Rebuild capsule pixels as ``(1-α)·T + α·C`` (thumbnail *T*, tint *C*) like real translucency.

    Solid median fills look like an opaque patch; this keeps underlying note texture visible.
    """

    import numpy as np

    arr_src = np.asarray(source_image.convert("RGB"), dtype=np.float32)
    patched_arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    ih, iw = arr_src.shape[:2]

    rx, ry = int(strip_roi["x"]), int(strip_roi["y"])
    rw, rh = int(strip_roi["width"]), int(strip_roi["height"])
    sx0, sy0 = int(thumb["x"]), int(thumb["y"])
    tw, th = int(thumb["width"]), int(thumb["height"])

    if rx + rw > iw or ry + rh > ih or sx0 + tw > iw or sy0 + th > ih:
        return inpaint_overlay_views_compact_fill(image, padded)

    strip_obs = arr_src[ry : ry + rh, rx : rx + rw]

    ty_rel = ry - sy0 - max(8, rh // 2)
    ty_rel = int(np.clip(ty_rel, 2, th - 3))

    baseline = np.zeros((rh, rw, 3), dtype=np.float32)
    for xi in range(rw):
        tx = rx + xi - sx0
        if tx < 1 or tx >= tw - 1:
            baseline[:, xi, :] = strip_obs[:, xi, :]
            continue
        patch = arr_src[
            sy0 + ty_rel - 2 : sy0 + ty_rel + 3,
            sx0 + tx - 1 : sx0 + tx + 2,
            :,
        ]
        med_col = np.median(patch.reshape(-1, 3), axis=0)
        baseline[:, xi, :] = med_col

    lum_med = np.median(
        0.299 * strip_obs[..., 0]
        + 0.587 * strip_obs[..., 1]
        + 0.114 * strip_obs[..., 2],
        axis=0,
    )

    ix_left = int(round(float(ink_rect["x"]) - rx))
    ix_right = int(round(float(ink_rect["x"] + ink_rect["width"]) - rx))

    interior_cols = [
        i
        for i in range(0, rw)
        if float(lum_med[i]) < 229.0 and (i < ix_left - 3 or i > ix_right + 3)
    ]

    if len(interior_cols) < 3:
        interior_cols = [
            i
            for i in range(0, rw)
            if float(lum_med[i]) < 235.0 and (i < ix_left - 2 or i > ix_right + 2)
        ]

    if len(interior_cols) < 3:
        return inpaint_overlay_views_compact_fill(image, padded)

    b_samples = baseline[:, interior_cols, :].reshape(-1, 3)
    o_samples = strip_obs[:, interior_cols, :].reshape(-1, 3)

    best: Optional[Tuple[float, float, np.ndarray]] = None
    for alpha in np.linspace(0.22, 0.55, 18):
        denom = float(alpha) if float(alpha) > 1e-3 else 1e-3
        d_samp = (o_samples - (1.0 - alpha) * b_samples) / denom
        d_samp = np.clip(d_samp, 0.0, 255.0)
        c_rgb = np.median(d_samp, axis=0)
        pred = (1.0 - alpha) * b_samples + alpha * c_rgb.reshape(1, 3)
        score = float(np.mean(np.abs(pred - o_samples)))
        if best is None or score < best[0]:
            best = (score, float(alpha), c_rgb.copy())

    if best is None:
        return inpaint_overlay_views_compact_fill(image, padded)

    alpha_f = best[1]
    c_rgb = np.clip(best[2], 0.0, 255.0)

    x0, y0, x1, y1 = rect_to_box(padded)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x1), min(ih, y1)

    for gy in range(y0, y1):
        ys = gy - ry
        if ys < 0 or ys >= rh:
            continue
        for gx in range(x0, x1):
            xs = gx - rx
            if xs < 0 or xs >= rw:
                continue
            bpx = baseline[ys, xs, :]
            patched_arr[gy, gx] = (1.0 - alpha_f) * bpx + alpha_f * c_rgb

    from PIL import Image

    return Image.fromarray(np.clip(np.round(patched_arr), 0, 255).astype(np.uint8))


def inpaint_overlay_views_stroke_fill(image, source_image, rect: Dict[str, int]):
    """Erase bright overlay strokes with reconstructed capsule material.

    The feed overlay is a semi-transparent grey capsule plus white icon/text.
    Copying nearby pixels can drag the eye icon or photo details into the old
    number slot. Instead, mask the old white strokes, rebuild just those pixels
    from clean capsule material, and feather the mask.
    """

    import numpy as np
    from PIL import Image, ImageChops, ImageFilter

    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    src = np.asarray(source_image.convert("RGB"), dtype=np.float32)
    ih, iw = arr.shape[:2]
    x0, y0, x1, y1 = rect_to_box(rect)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x1), min(ih, y1)
    if x1 <= x0 or y1 <= y0:
        return image

    crop = src[y0:y1, x0:x1, :]
    lum = 0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]
    spread = crop.max(axis=2) - crop.min(axis=2)
    bg_lum = float(np.percentile(lum, 45))
    core_mask = (lum > max(205.0, bg_lum + 64.0)) & (spread < 110.0)
    halo_mask = (lum > max(158.0, bg_lum + 34.0)) & (spread < 130.0)
    # Use the softer mask to catch compressed antialias fragments. If it starts
    # swallowing the capsule/background, fall back to the bright core.
    mask = halo_mask if int(halo_mask.sum()) >= int(core_mask.sum()) else core_mask
    if float(mask.mean()) > 0.42 and int(core_mask.sum()) >= 2:
        mask = core_mask
    if int(mask.sum()) < 2:
        return image

    seen = np.zeros(mask.shape, dtype=bool)
    components = []
    mh, mw = mask.shape
    for yy in range(mh):
        for xx in range(mw):
            if not mask[yy, xx] or seen[yy, xx]:
                continue
            stack = [(xx, yy)]
            seen[yy, xx] = True
            pts = []
            while stack:
                px, py = stack.pop()
                pts.append((px, py))
                for ny in range(py - 1, py + 2):
                    for nx in range(px - 1, px + 2):
                        if 0 <= nx < mw and 0 <= ny < mh and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            if pts:
                components.append(pts)
    if components:
        plausible = []
        for pts in components:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            comp_w = max(xs) - min(xs) + 1
            comp_h = max(ys) - min(ys) + 1
            comp_area = len(pts)
            touches_right_edge = max(xs) >= mw - 2
            touches_vertical_edge = min(ys) <= 1 or max(ys) >= mh - 2
            too_wide = comp_w > max(12, int(round(mw * 0.72)))
            too_tall = comp_h > max(12, int(round(mh * 0.88)))
            if too_wide or too_tall:
                continue
            if touches_right_edge and touches_vertical_edge:
                continue
            plausible.append(pts)
        kept = plausible or sorted(components, key=len, reverse=True)[: max(1, min(4, len(components)))]
        clean_mask = np.zeros(mask.shape, dtype=bool)
        for pts in kept:
            for px, py in pts:
                clean_mask[py, px] = True
        mask = clean_mask

    ys, xs = np.where(mask)
    stroke_bounds = None
    stroke_mask_for_inpaint = mask.copy()
    if xs.size:
        stroke_bounds = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        pad_x = 3 if mw > 18 else (2 if mw > 14 else 1)
        pad_y = 2 if mh >= 14 else 1
        bx0 = max(0, int(xs.min()) - pad_x)
        bx1 = min(mw - 1, int(xs.max()) + pad_x)
        by0 = max(0, int(ys.min()) - pad_y)
        by1 = min(mh - 1, int(ys.max()) + pad_y)
        envelope_area = float((bx1 - bx0 + 1) * (by1 - by0 + 1))
        rect_area = float(max(1, mw * mh))
        if envelope_area / rect_area <= 0.72:
            envelope = np.zeros(mask.shape, dtype=bool)
            envelope[by0 : by1 + 1, bx0 : bx1 + 1] = True
            mask = envelope

    if stroke_bounds is not None and min(mw, mh) <= 26:
        # For tiny feed-overlay digits, diffusion from the same crop can pull
        # the old anti-aliased edge back into the fill. Inpaint the dilated
        # stroke mask directly, not a bounding rectangle, so capsule corners
        # and edge highlights outside the glyph strokes are preserved.
        try:
            import cv2

            filter_size = 5 if min(mw, mh) <= 22 else 7
            local_mask_img = Image.fromarray(
                (stroke_mask_for_inpaint.astype(np.uint8) * 255), mode="L"
            ).filter(ImageFilter.MaxFilter(filter_size))
            local_mask = np.asarray(local_mask_img, dtype=np.uint8)
            if int((local_mask > 0).sum()) >= 2:
                source_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
                cv_mask = np.zeros((ih, iw), dtype=np.uint8)
                cv_mask[y0:y1, x0:x1] = local_mask
                bgr = cv2.cvtColor(source_arr, cv2.COLOR_RGB2BGR)
                repaired = cv2.inpaint(bgr, cv_mask, 2, cv2.INPAINT_TELEA)
                return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))
        except Exception:
            pass

    if min(mw, mh) <= 14:
        max_filter_size, blur_radius = 3, 1.2
    elif min(mw, mh) <= 22:
        max_filter_size, blur_radius = 5, 1.0
    else:
        max_filter_size, blur_radius = 7, 0.8

    hard_mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.MaxFilter(max_filter_size)
    )
    repair_mask = np.asarray(hard_mask_img, dtype=np.uint8) > 0
    # Keep the detected old strokes fully opaque in the erase mask. A purely
    # blurred mask blends a little of the original white antialiasing back in,
    # which shows up as dirty remnants under the replacement number.
    soft_mask_img = hard_mask_img.filter(ImageFilter.GaussianBlur(blur_radius))
    mask_img = ImageChops.lighter(hard_mask_img, soft_mask_img)

    fill = crop.copy()
    unknown = repair_mask.copy()
    known = ~unknown
    if not known.any():
        return image

    # Reconstruct old white strokes from immediately adjacent capsule pixels.
    # The capsule is translucent over arbitrary photos; local diffusion preserves
    # wood grain/highlight gradients much better than a single median fill.
    for _ in range(max(8, mw + mh)):
        if not unknown.any():
            break
        new_fill = fill.copy()
        new_known = known.copy()
        ys, xs = np.where(unknown)
        for yy, xx in zip(ys.tolist(), xs.tolist()):
            samples = []
            for ny in range(max(0, yy - 1), min(mh, yy + 2)):
                for nx in range(max(0, xx - 1), min(mw, xx + 2)):
                    if ny == yy and nx == xx:
                        continue
                    if known[ny, nx]:
                        samples.append(fill[ny, nx])
            if samples:
                new_fill[yy, xx] = np.mean(np.asarray(samples, dtype=np.float32), axis=0)
                new_known[yy, xx] = True
        if np.array_equal(new_known, known):
            break
        fill = new_fill
        known = new_known
        unknown = ~known

    if unknown.any():
        known_pixels = fill[known]
        if known_pixels.shape[0] < 1:
            return image
        fill[unknown] = np.median(known_pixels, axis=0)

    fill_img = Image.fromarray(np.clip(np.round(fill), 0, 255).astype(np.uint8))
    patched = Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8)).convert("RGBA")
    patched.paste(fill_img.convert("RGBA"), (x0, y0), mask_img)

    return patched.convert("RGB")


def load_font(size: int, weight: str):
    from PIL import ImageFont

    from .red_number_fonts import path_for_pil_weight

    bundled = path_for_pil_weight(weight if weight in ("bold", "medium", "regular", "normal") else "medium")
    if bundled is not None:
        try:
            return ImageFont.truetype(str(bundled), size=size)
        except Exception:
            pass

    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if weight == "bold" else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf" if weight in ("regular", "medium") else "",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if weight == "bold" else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_font_by_path(path: str, size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return None


def iter_candidate_fonts(size: int):
    """Fonts tried during RMSE calibration. Prefer bundled RED Number only when present."""

    from .red_number_fonts import bundled_font_paths_in_order

    yielded = False
    bundled = bundled_font_paths_in_order()
    paths = bundled if bundled else FONT_CANDIDATE_PATHS
    for path in paths:
        if not Path(path).exists():
            continue
        font = load_font_by_path(path, size)
        if font is not None:
            yielded = True
            yield path, font
    if not yielded:
        yield "default-bold", load_font(size, "bold")


def rendered_ink_bbox(text: str, font):
    from PIL import Image, ImageDraw

    origin = (40, 40)
    image = Image.new("L", (360, 120), 0)
    draw = ImageDraw.Draw(image)
    draw.text(origin, text, fill=255, font=font)
    bbox = image.getbbox()
    if bbox is None:
        return (0, 0, 0, 0)
    return (
        bbox[0] - origin[0],
        bbox[1] - origin[1],
        bbox[2] - origin[0],
        bbox[3] - origin[1],
    )


def _metric_text_segments(text: str) -> list[str]:
    return [p for p in re.split(r"(%)", str(text)) if p]


def _weight_from_font_path(font_path: Optional[str]) -> str:
    if not font_path:
        return "medium"
    lower = str(font_path).lower()
    if "bold" in lower:
        return "bold"
    return "medium"


def load_font_symbol_fallback(size: int, weight: str):
    """Fonts that include ``%`` (U+0025). RED Number omits it — PIL draws a black tofu otherwise."""

    from PIL import ImageFont

    paths: List[Tuple[str, Optional[int]]] = [
        ("/System/Library/Fonts/PingFang.ttc", 0),
    ]
    if weight == "bold":
        paths.append(("/System/Library/Fonts/Supplemental/Arial Bold.ttf", None))
    else:
        paths.append(("/System/Library/Fonts/Supplemental/Arial.ttf", None))
    paths.append(("C:/Windows/Fonts/msyh.ttc", None))
    if weight == "bold":
        paths.append(("C:/Windows/Fonts/arialbd.ttf", None))
    else:
        paths.append(("C:/Windows/Fonts/arial.ttf", None))
    for path, idx in paths:
        if not path or not Path(path).exists():
            continue
        try:
            if idx is not None:
                return ImageFont.truetype(path, size=size, index=idx)
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_percent_glyph_font(size: int, weight: str):
    """Percent glyph only: bundled DIN-OT-Medium (has U+0025), pairs with RED Number digits."""

    from PIL import ImageFont

    eff = int(size)
    if weight == "bold":
        eff = max(eff, int(round(size * 1.04)))

    if BUNDLED_DIN_PERCENT.is_file():
        try:
            return ImageFont.truetype(str(BUNDLED_DIN_PERCENT), size=eff)
        except Exception:
            pass
    return load_font_symbol_fallback(size, weight)


def load_red_metric_segment_font(size: int, weight: str):
    """Digits / dot use bundled RED Number when available."""

    from PIL import ImageFont

    p = path_for_pil_weight(weight)
    if p is not None and p.is_file():
        try:
            return ImageFont.truetype(str(p), size=size)
        except Exception:
            pass
    return load_font_symbol_fallback(size, weight)


def _use_red_percent_hybrid(
    text: str,
    *,
    font=None,
    font_path_hint: Optional[str] = None,
) -> bool:
    """Paint ``%`` with DIN bundle (or symbol fallback) when digits use RED Number."""

    if "%" not in str(text) or not RED_NUMBER_BOLD.is_file():
        return False
    if font_path_hint and "REDNumber" in str(font_path_hint):
        return True
    p = getattr(font, "path", None)
    return bool(p and "REDNumber" in str(p))


def _advance_text_cursor(draw, x: int, y: int, part: str, seg_font) -> int:
    try:
        bb = draw.textbbox((x, y), part, font=seg_font)
        return int(bb[2])
    except Exception:
        bb = rendered_ink_bbox(part, seg_font)
        return x + max(0, bb[2] - bb[0])


@lru_cache(maxsize=128)
def _percent_vertical_nudge_px(size: int, weight: str) -> int:
    """Extra Y (down) for the ``%`` glyph so DIN/回退字体与 RED 数字视觉中线对齐。

    默认锚点下 % 的 ink 往往偏上；按与「8」的 ink 中心差自动下移。
    """

    from PIL import Image, ImageDraw

    if size < 1:
        return 0
    try:
        rf = load_red_metric_segment_font(size, weight)
        pf = load_percent_glyph_font(size, weight)
    except Exception:
        return max(0, int(round(size * 0.06)))

    ox, oy = 40, 60
    im8 = Image.new("L", (240, 180), 0)
    ImageDraw.Draw(im8).text((ox, oy), "8", font=rf, fill=255)
    bb8 = im8.getbbox()
    imp = Image.new("L", (240, 180), 0)
    ImageDraw.Draw(imp).text((ox, oy), "%", font=pf, fill=255)
    bbp = imp.getbbox()
    if not bb8 or not bbp:
        return max(0, int(round(size * 0.06)))
    c8 = (bb8[1] + bb8[3]) / 2.0
    cp = (bbp[1] + bbp[3]) / 2.0
    return int(round(c8 - cp))


def draw_metric_red_plus_percent(draw, origin_xy: Tuple[int, int], text: str, size: int, weight: str, fill) -> None:
    x0, y0 = origin_xy
    x = int(x0)
    nudge_y = _percent_vertical_nudge_px(size, weight)
    for part in _metric_text_segments(text):
        seg_font = (
            load_percent_glyph_font(size, weight) if part == "%" else load_red_metric_segment_font(size, weight)
        )
        y_part = y0 + (nudge_y if part == "%" else 0)
        draw.text((x, y_part), part, font=seg_font, fill=fill)
        x = _advance_text_cursor(draw, x, y_part, part, seg_font)


def rendered_ink_bbox_red_percent_split(text: str, size: int, weight: str) -> Tuple[int, int, int, int]:
    from PIL import Image, ImageDraw

    origin = (40, 40)
    image = Image.new("L", (520, 160), 0)
    draw = ImageDraw.Draw(image)
    draw_metric_red_plus_percent(draw, origin, text, size, weight, 255)
    bbox = image.getbbox()
    if bbox is None:
        return (0, 0, 0, 0)
    return (
        bbox[0] - origin[0],
        bbox[1] - origin[1],
        bbox[2] - origin[0],
        bbox[3] - origin[1],
    )


def rendered_ink_stats_red_percent_split(text: str, size: int, weight: str):
    from PIL import Image, ImageDraw

    origin = (40, 40)
    image = Image.new("L", (520, 160), 0)
    draw = ImageDraw.Draw(image)
    draw_metric_red_plus_percent(draw, origin, text, size, weight, 255)
    bbox = image.getbbox()
    if bbox is None:
        return {"bbox": (0, 0, 0, 0), "density": 0}
    crop = image.crop(bbox)
    values = list(crop.getdata())
    ink_count = sum(1 for value in values if value > 24)
    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return {
        "bbox": (
            bbox[0] - origin[0],
            bbox[1] - origin[1],
            bbox[2] - origin[0],
            bbox[3] - origin[1],
        ),
        "density": ink_count / area,
    }


def _overlay_views_left_nudge_px(original_text: str, new_text: str, *, font) -> int:
    """Negative horizontal delta when ``new_text`` renders wider than ``original_text``.

    Calibration fits offsets using the original string; a wider replacement otherwise sits
    too far right relative to the eye icon / capsule. Shift slightly left in proportion
    to extra pixel width (capped).
    """

    if font is None or not original_text or not new_text:
        return 0
    bo = rendered_ink_bbox(str(original_text), font)
    bn = rendered_ink_bbox(str(new_text), font)
    wo = max(0, bo[2] - bo[0])
    wn = max(0, bn[2] - bn[0])
    extra = wn - wo
    if extra <= 0:
        return 0
    h_ref = max(1, bo[3] - bo[1], bn[3] - bn[1])
    cap = max(4, min(10, int(round(h_ref * 0.32))))
    return -min(max(1, round(extra * 0.34)), cap)


def _feed_overlay_visual_dx(original_text: str, new_text: str) -> int:
    """Position tiny feed overlay digits by the source slot, not one global offset."""

    old_core = re.sub(r"[^0-9.]", "", normalize_metric_text(original_text))
    new_core = re.sub(r"[^0-9.]", "", normalize_metric_text(new_text))
    if len(old_core) >= 2 and len(new_core) >= 2:
        return -1
    return FEED_OVERLAY_VIEWS_DX


def _feed_overlay_visual_dy(original_text: str, new_text: str) -> int:
    old_core = re.sub(r"[^0-9.]", "", normalize_metric_text(original_text))
    new_core = re.sub(r"[^0-9.]", "", normalize_metric_text(new_text))
    if len(old_core) >= 2 and len(new_core) >= 2:
        return 0
    return FEED_OVERLAY_VIEWS_DY


def rendered_ink_stats(text: str, font):
    from PIL import Image, ImageDraw

    origin = (40, 40)
    image = Image.new("L", (420, 140), 0)
    draw = ImageDraw.Draw(image)
    draw.text(origin, text, fill=255, font=font)
    bbox = image.getbbox()
    if bbox is None:
        return {"bbox": (0, 0, 0, 0), "density": 0}
    crop = image.crop(bbox)
    values = list(crop.getdata())
    ink_count = sum(1 for value in values if value > 24)
    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return {
        "bbox": (
            bbox[0] - origin[0],
            bbox[1] - origin[1],
            bbox[2] - origin[0],
            bbox[3] - origin[1],
        ),
        "density": ink_count / area,
    }


def mask_density(mask):
    bbox = mask.getbbox()
    if bbox is None:
        return 0
    crop = mask.crop(bbox)
    values = list(crop.getdata())
    ink_count = sum(1 for value in values if value > 24)
    area = max(1, crop.width * crop.height)
    return ink_count / area


def summarize_values(values):
    if not values:
        return {"p10": 0, "p50": 0, "p90": 0}
    ordered = sorted(values)
    last = len(ordered) - 1
    return {
        "p10": ordered[round(last * 0.1)],
        "p50": ordered[round(last * 0.5)],
        "p90": ordered[round(last * 0.9)],
    }


def mask_style(mask):
    import numpy as np

    bbox = mask.getbbox()
    if bbox is None:
        return {"density": 0, "edge_ratio": 0, "alpha_summary": {"p10": 0, "p50": 0, "p90": 0}}
    crop = mask.crop(bbox)
    arr = np.array(crop, dtype=np.uint8)
    area = max(1, arr.size)
    ink = arr[arr > 0]
    if ink.size == 0:
        return {"density": 0, "edge_ratio": 0, "alpha_summary": {"p10": 0, "p50": 0, "p90": 0}}
    core_count = int((ink >= 220).sum())
    edge_count = int(ink.size - core_count)
    total = max(1, core_count + edge_count)
    values = ink.tolist()
    return {
        "density": ink.size / area,
        "edge_ratio": edge_count / total,
        "alpha_summary": summarize_values(values),
    }


def style_distance(target, candidate):
    target_alpha = target["alpha_summary"]
    candidate_alpha = candidate["alpha_summary"]
    alpha_score = (
        abs(target_alpha["p10"] - candidate_alpha["p10"])
        + abs(target_alpha["p50"] - candidate_alpha["p50"])
        + abs(target_alpha["p90"] - candidate_alpha["p90"])
    ) / 255
    density_score = abs(target["density"] - candidate["density"]) * 12
    edge_score = abs(target["edge_ratio"] - candidate["edge_ratio"]) * 6
    return round(alpha_score + density_score + edge_score, 6)


def tune_mask_weight(mask, target_density: float):
    from PIL import ImageChops, ImageFilter

    candidates = [mask]
    shifted_right = ImageChops.offset(mask, 1, 0)
    shifted_down = ImageChops.offset(mask, 0, 1)
    candidates.append(ImageChops.lighter(mask, shifted_right))
    candidates.append(ImageChops.lighter(mask, shifted_down))
    # Only thicken when needed. Thinning small screenshot text makes it look washed out and
    # destroys the hard compressed edge we want to preserve.
    if mask_density(mask) < target_density * 0.9:
        thicker = mask.filter(ImageFilter.MaxFilter(3))
        candidates.append(thicker)
        candidates.append(thicker.filter(ImageFilter.MaxFilter(3)))

    return min(candidates, key=lambda candidate: abs(mask_density(candidate) - target_density))


def weight_mask_variants(mask):
    from PIL import ImageChops, ImageFilter

    shifted_right = ImageChops.offset(mask, 1, 0)
    shifted_down = ImageChops.offset(mask, 0, 1)
    embolden_x = ImageChops.lighter(mask, shifted_right)
    embolden_y = ImageChops.lighter(mask, shifted_down)
    embolden_xy = ImageChops.lighter(embolden_x, shifted_down)
    embolden_full = mask.filter(ImageFilter.MaxFilter(3))

    return [
        ("w0", mask),
        ("w1x", embolden_x),
        ("w1y", embolden_y),
        ("w1xy", embolden_xy),
        ("w2", embolden_full),
    ]


def fit_font_to_ink(text: str, target_rect: Dict[str, int], target_density: float):
    """Return ``(font, ink_bbox, weight)``. Hybrid RED+% uses RED cmap-safe sizing."""

    target_height = max(1, target_rect["height"])
    best = None

    if "%" in str(text) and RED_NUMBER_BOLD.is_file():
        for size in range(8, 72):
            for weight in ("regular", "medium", "bold"):
                stats = rendered_ink_stats_red_percent_split(str(text), size, weight)
                bbox = stats["bbox"]
                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                height_score = abs(height - target_height) * 2.0
                density_score = abs(stats["density"] - target_density) * 18.0
                width_score = abs(width - target_rect["width"]) * 0.04
                score = height_score + density_score + width_score
                if best is None or score < best[0]:
                    font = load_red_metric_segment_font(size, weight)
                    best = (score, font, bbox, weight)

        if best is None:
            fb = load_percent_glyph_font(max(8, target_height), "bold")
            return fb, rendered_ink_bbox(str(text), fb), "bold"

        return best[1], best[2], best[3]

    for size in range(8, 72):
        for weight in ("regular", "medium", "bold"):
            font = load_font(size, weight)
            stats = rendered_ink_stats(str(text), font)
            bbox = stats["bbox"]
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            height_score = abs(height - target_height) * 2.0
            density_score = abs(stats["density"] - target_density) * 18.0
            width_score = abs(width - target_rect["width"]) * 0.04
            score = height_score + density_score + width_score
            if best is None or score < best[0]:
                best = (score, font, bbox, weight)

    if best is None:
        font = load_font(target_height, "bold")
        return font, rendered_ink_bbox(str(text), font), "bold"

    return best[1], best[2], best[3]


def choose_font_size_for_rendered_height(font_path: str, texts, target_height: int) -> int:
    """Choose a font size whose rendered ink height matches ``target_height``.

    PIL font size is not the same as the visible glyph height.  A fixed
    multiplier such as ``ink_height * 1.2`` can look right for one screenshot
    but too small for another.  Measure the rendered bbox for the actual
    replacement texts and pick the size whose median visible height best
    matches the source row.
    """
    target_height = max(1, int(target_height))
    weight = _weight_from_font_path(font_path)
    best = None
    for size in range(max(8, target_height), min(96, target_height * 2 + 24) + 1):
        font = load_font_by_path(font_path, size)
        if font is None:
            font = load_font(size, "bold")
        heights = []
        for text in texts:
            ts = str(text)
            if "%" in ts and _use_red_percent_hybrid(ts, font_path_hint=str(font_path), font=font):
                bbox = rendered_ink_bbox_red_percent_split(ts, size, weight)
            else:
                bbox = rendered_ink_bbox(ts, font)
            height = bbox[3] - bbox[1]
            if height > 0:
                heights.append(height)
        if not heights:
            continue
        median_height = sorted(heights)[len(heights) // 2]
        score = abs(median_height - target_height)
        if best is None or score < best[0]:
            best = (score, size)
    return best[1] if best else target_height


def choose_feed_overlay_font_size(font_path: str, texts, ink_rect: Dict[str, int]) -> int:
    """Fit tiny feed overlay digits by natural rendered size.

    Callers should pass the source/original digits when replacing text.  That
    makes the chosen font size a calibration against what is already in the
    screenshot; the replacement then uses the same natural RED Number size.
    """

    ink_h = max(1, int(ink_rect["height"]))
    ink_w = max(1, int(ink_rect["width"]))
    text_list = [str(text) for text in texts]
    est_digit_count = max(1, max((sum(1 for ch in text if ch.isdigit()) for text in text_list), default=1))
    if ink_h <= 14:
        # OCR/localized overlay slots include a little antialias halo and nearby
        # cover texture. Prefer the natural RED Number size that best recreates
        # the old digit slot; replacement digits are rendered at that size.
        target_height = max(8, ink_h - 1)
        target_width = max(6.0, float(ink_w))
        original_digits = "".join(ch for ch in text_list[0] if ch.isdigit()) if text_list else ""
        replacement_digits = "".join(ch for ch in text_list[-1] if ch.isdigit()) if text_list else ""
        if original_digits and set(original_digits) <= {"1"} and set(replacement_digits) - {"1"}:
            target_width = max(target_width, est_digit_count * target_height * 0.72)
    else:
        # PaddleOCR can return a loose rectangle that covers the translucent
        # capsule/background, not just the tiny white strokes. When the visual
        # localizer cannot tighten it, sizing directly to that outer box makes
        # two-digit counts look inflated. Use a conservative per-digit slot
        # estimate for larger overlay boxes.
        target_height = max(8, int(round(ink_h * 0.66)))
        target_width = max(6.0, min(float(ink_w), est_digit_count * target_height * 0.72))
    weight = _weight_from_font_path(font_path)
    best = None
    for size in range(8, min(48, max(12, int(ink_rect["height"]) * 2 + 12)) + 1):
        font = load_font_by_path(font_path, size)
        if font is None:
            font = load_font(size, "medium")
        heights = []
        widths = []
        for ts in text_list:
            if "%" in ts and _use_red_percent_hybrid(ts, font_path_hint=str(font_path), font=font):
                bbox = rendered_ink_bbox_red_percent_split(ts, size, weight)
            else:
                bbox = rendered_ink_bbox(ts, font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
        if not heights:
            continue
        med_h = sorted(heights)[len(heights) // 2]
        max_w = max(widths)
        width_overflow = max(0.0, max_w - target_width)
        height_overflow = max(0.0, med_h - target_height)
        score = (
            width_overflow * 9.0
            + height_overflow * 5.0
            + abs(med_h - target_height) * 1.8
            + abs(max_w - target_width) * 0.35
        )
        if best is None or score < best[0]:
            best = (score, size)
    return best[1] if best else target_height


def red_number_forced_font_for_standalone_patch(
    *,
    ink_rect: Dict[str, int],
    original_text: str,
    new_text: str,
    overlay_views_ink: bool,
) -> Optional[Dict]:
    """Force bundled RED Number for single-field patches (顶部观看数、信息流小眼睛叠加).

    详情区多块指标已在 ``beautify_normal_with_ocr`` 里用 ``body_forced_font``；此处对齐同一字形。
    封面浅色叠加不套用 ``BODY_NATIVE_FORCE_EDGE_VARIANT``，便于按浅色笔画匹配边缘。
    """

    if not prefer_red_number_metric_render():
        return None
    ot = normalize_metric_text(original_text)
    nt = normalize_metric_text(str(new_text))
    if not is_pure_metric_text(nt) or not ot:
        return None
    if not any(ch.isdigit() for ch in ot):
        return None

    if overlay_views_ink:
        candidates = (
            BUNDLED_FEED_OVERLAY_VIEWS_FONT,
            BUNDLED_DIN_PERCENT,
            RED_NUMBER_REGULAR,
            RED_NUMBER_MEDIUM,
            RED_NUMBER_BOLD,
        )
        fp = next((str(path) for path in candidates if path.is_file()), "")
    else:
        fp = str(BODY_NATIVE_FONT_PATH)
    if not fp or not Path(fp).is_file():
        return None

    target_h = max(8, int(ink_rect["height"]))
    texts_for_size = list(dict.fromkeys([ot, nt]))
    if overlay_views_ink:
        font_size = choose_feed_overlay_font_size(fp, texts_for_size, ink_rect)
    else:
        font_size = max(
            8,
            int(choose_font_size_for_rendered_height(fp, texts_for_size, target_h))
            + BODY_NATIVE_FONT_SIZE_ADJUST,
        )
    out: Dict = {
        "font_size": font_size,
        "font_path": fp,
        "font_match": {
            "target_height": target_h,
            "matched_chars": len(nt),
            "standalone_red_patch": True,
        },
    }
    if not overlay_views_ink:
        if BODY_NATIVE_FORCE_EDGE_VARIANT is not None:
            out["force_edge_variant"] = BODY_NATIVE_FORCE_EDGE_VARIANT
        if BODY_NATIVE_FORCE_ALPHA_STRENGTH is not None:
            out["force_alpha_match_strength"] = BODY_NATIVE_FORCE_ALPHA_STRENGTH
    else:
        out["overlay_visual_dx"] = _feed_overlay_visual_dx(ot, nt)
        out["overlay_visual_dy"] = _feed_overlay_visual_dy(ot, nt)
        out["overlay_alpha_gain"] = FEED_OVERLAY_VIEWS_ALPHA_GAIN
        out["overlay_alpha_gamma"] = FEED_OVERLAY_VIEWS_ALPHA_GAMMA
        out["overlay_color"] = (255, 255, 255)
    return out


def build_header_views_forced_font(
    *,
    ink_rect: Dict[str, int],
    original_text: str,
    new_text: str,
) -> Optional[Dict]:
    """顶部「小眼睛」旁数字：RED Number Medium / Regular，匹配小号灰色正文而非详情粗重红数字。"""

    if not prefer_red_number_metric_render():
        return None
    ot = normalize_metric_text(original_text)
    nt = normalize_metric_text(str(new_text))
    if not is_pure_metric_text(nt) or not ot or not any(ch.isdigit() for ch in ot):
        return None

    fp: Optional[str] = None
    for candidate in (RED_NUMBER_REGULAR, RED_NUMBER_MEDIUM, RED_NUMBER_BOLD):
        if candidate.is_file():
            fp = str(candidate)
            break
    if not fp:
        return None

    target_h = max(8, int(ink_rect["height"]))
    texts_for_size = list(dict.fromkeys([ot, nt]))
    chosen = choose_font_size_for_rendered_height(fp, texts_for_size, target_h)
    # 顶部栏视觉高度≈12–14px：PIL 常用字号易偏大，按墨迹高度收紧上限。
    stretch = 1.06 + 0.05 * max(0, len(nt) - 2)
    cap_px = max(10, int(round(target_h * stretch)))
    font_size = max(8, min(chosen + HEADER_VIEWS_FONT_ADJUST, cap_px))
    return {
        "font_size": font_size,
        "font_path": fp,
        "font_match": {
            "target_height": target_h,
            "matched_chars": len(nt),
            "header_views": True,
        },
    }


def _median(values) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def choose_best_body_font(texts, source_stats: Dict[str, object]) -> Dict[str, object]:
    """Pick font for body metric digits. Prefer bundled RED Number over geometry-matched system fonts."""

    target_height = int(source_stats.get("target_height") or 1)
    if RED_NUMBER_BOLD.is_file():
        fp = str(RED_NUMBER_BOLD)
        return {
            "font_path": fp,
            "font_size": choose_font_size_for_rendered_height(fp, texts, target_height),
            "score": 0.0,
            "matched_chars": len(source_stats.get("char_widths") or {}),
            "target_height": target_height,
        }

    target_density = float(source_stats.get("target_density") or 0)
    target_edge_ratio = float(source_stats.get("target_edge_ratio") or 0)
    char_widths = source_stats.get("char_widths") or {}
    best = None

    for font_path in BODY_NATIVE_FONT_CANDIDATE_PATHS:
        if not Path(font_path).exists():
            continue
        font_size = choose_font_size_for_rendered_height(font_path, texts, target_height)
        font = load_font_by_path(font_path, font_size)
        if font is None:
            continue

        heights = []
        for text in texts:
            ts = str(text)
            if "%" in ts and _use_red_percent_hybrid(ts, font_path_hint=str(font_path), font=font):
                bbox = rendered_ink_bbox_red_percent_split(ts, font_size, _weight_from_font_path(font_path))
            else:
                bbox = rendered_ink_bbox(ts, font)
            heights.append(bbox[3] - bbox[1])
        score = abs(_median(heights) - target_height) * 1.25

        matched_chars = 0
        for char, widths in char_widths.items():
            if not widths:
                continue
            if char == "%":
                fb = load_percent_glyph_font(font_size, _weight_from_font_path(font_path))
                bbox = rendered_ink_bbox(str(char), fb)
            else:
                bbox = rendered_ink_bbox(str(char), font)
            rendered_width = bbox[2] - bbox[0]
            source_width = _median(widths)
            score += abs(rendered_width - source_width) * 2.0
            matched_chars += 1

        if target_density > 0:
            styles = []
            for text in texts:
                ts = str(text)
                if "%" in ts and _use_red_percent_hybrid(ts, font_path_hint=str(font_path), font=font):
                    bbox = rendered_ink_bbox_red_percent_split(ts, font_size, _weight_from_font_path(font_path))
                else:
                    bbox = rendered_ink_bbox(ts, font)
                mask = text_mask_for_candidate(
                    (420, 160),
                    ts,
                    font,
                    (40 - bbox[0], 40 - bbox[1]),
                    font_path_hint=str(font_path),
                )
                target_style = {
                    "density": target_density,
                    "edge_ratio": target_edge_ratio,
                    "alpha_summary": {"p10": 255, "p50": 255, "p90": 255},
                }
                best_style = None
                for variant_name, strength, candidate_mask in candidate_masks(
                    mask,
                    {
                        "alpha_values": [255],
                        "density": target_density,
                        "edge_ratio": 0,
                        "alpha_summary": {"p10": 255, "p50": 255, "p90": 255},
                    },
                ):
                    if (
                        BODY_NATIVE_FORCE_EDGE_VARIANT is not None
                        and variant_name != BODY_NATIVE_FORCE_EDGE_VARIANT
                    ):
                        continue
                    if (
                        BODY_NATIVE_FORCE_ALPHA_STRENGTH is not None
                        and strength != BODY_NATIVE_FORCE_ALPHA_STRENGTH
                    ):
                        continue
                    candidate_style = mask_style(candidate_mask)
                    candidate_score = style_distance(target_style, candidate_style)
                    if best_style is None or candidate_score < best_style[0]:
                        best_style = (candidate_score, candidate_style)
                if best_style is not None:
                    styles.append(best_style[1])
            if styles:
                score += abs(_median([style["density"] for style in styles]) - target_density) * 100.0
                score += abs(_median([style["edge_ratio"] for style in styles]) - target_edge_ratio) * 25.0

        # Prefer fonts that can explain more sampled characters.
        score -= matched_chars * 0.05
        if best is None or score < best["score"]:
            best = {
                "font_path": font_path,
                "font_size": font_size,
                "score": score,
                "matched_chars": matched_chars,
                "target_height": target_height,
            }

    if best is not None:
        return best
    return {
        "font_path": BODY_NATIVE_FONT_PATH,
        "font_size": choose_font_size_for_rendered_height(BODY_NATIVE_FONT_PATH, texts, target_height),
        "score": None,
        "matched_chars": 0,
        "target_height": target_height,
    }


def locate_text_rect(image, field_config: Dict[str, object], use_ocr: bool = False) -> Dict[str, int]:
    image_size = {"width": image.width, "height": image.height}
    fallback = scale_rect(field_config["fallback"], BASE_SIZE, image_size)  # type: ignore[arg-type]
    search = scale_rect(field_config["search"], BASE_SIZE, image_size)  # type: ignore[arg-type]

    if use_ocr:
        cv2, np = load_optional_cv2()
        if np is None:
            raise RuntimeError("OCR mode requires numpy. Install requirements.txt.")
        crop = image.crop(rect_to_box(search)).convert("RGB")
        bounds = detect_bounds_with_paddle(np.array(crop))
        if bounds:
            detected = translate_rect(bounds, search)
            return expand_rect(detected, max(3, detected["height"] // 6), image_size)

    if field_config.get("prefer_template", True):
        return fallback

    bounds = detect_dark_text_bounds(crop_pixels(image, search))
    if not bounds:
        return fallback
    detected = translate_rect(bounds, search)
    if detected["height"] < fallback["height"] * 0.55 or detected["width"] < fallback["width"] * 0.2:
        return fallback
    return expand_rect(detected, max(4, detected["height"] // 4), image_size)


def draw_text(image, rect: Dict[str, int], text: str, field_config: Dict[str, object]):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    scale = min(image.width / BASE_SIZE["width"], image.height / BASE_SIZE["height"])
    font_size = max(10, round(int(field_config["font_size"]) * scale))
    font = load_font(font_size, str(field_config.get("font_weight", "medium")))
    text_point = scale_rect(
        {
            "x": field_config["text"]["x"],
            "y": field_config["text"]["y"],
            "width": 1,
            "height": 1,
        },
        BASE_SIZE,
        {"width": image.width, "height": image.height},
    )
    position = (text_point["x"], text_point["y"] - font_size)
    weight = str(field_config.get("font_weight", "medium"))
    red_hint = path_for_pil_weight(weight)
    if _use_red_percent_hybrid(str(text), font_path_hint=str(red_hint) if red_hint else ""):
        draw_metric_red_plus_percent(draw, position, str(text), font_size, weight, (34, 34, 34))
    else:
        draw.text(position, text, fill=(34, 34, 34), font=font)
    return image


def extract_ink_color(image, rect: Dict[str, int]):
    import numpy as np

    arr = np.array(image.crop(rect_to_box(rect)).convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return (34, 34, 34)
    pixels = arr.reshape(-1, 3)
    luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    is_ink = (luminance < 180) & ((pixels.max(axis=1) - pixels.min(axis=1)) < 90)
    if not is_ink.any():
        return (34, 34, 34)
    ink = pixels[is_ink]
    ink_lum = 0.299 * ink[:, 0] + 0.587 * ink[:, 1] + 0.114 * ink[:, 2]
    order = np.argsort(ink_lum)
    core = ink[order[: max(1, int(len(order) * 0.65))]]
    mean = core.mean(axis=0)
    return (round(float(mean[0])), round(float(mean[1])), round(float(mean[2])))


def extract_light_overlay_text_color(image, rect: Dict[str, int]):
    """Sample RGB for bright overlay strokes (e.g. white view count on cover pill).

    ``extract_ink_color`` only considers dark pixels (body text); thumbnail overlays
    are often near-white. Returns ``None`` when no confident bright-neutral pixels.
    """

    import numpy as np

    arr = np.array(image.crop(rect_to_box(rect)).convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return None
    pixels = arr.reshape(-1, 3)
    lum = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    mask = (lum >= 232.0) & (chroma < 52.0)
    if int(mask.sum()) < 4:
        mask = (lum >= 218.0) & (chroma < 62.0)
    if int(mask.sum()) < 4:
        return None
    sel = pixels[mask]
    sel_lum = lum[mask]
    order = np.argsort(-sel_lum)
    k = max(4, int(len(order) * 0.5))
    core = sel[order[:k]]
    mean = core.mean(axis=0)
    return (round(float(mean[0])), round(float(mean[1])), round(float(mean[2])))


def extract_light_overlay_ink_style(image, rect: Dict[str, int]):
    """Estimate color, alpha and edge style for tiny light feed-overlay digits."""

    import numpy as np

    arr = np.array(image.crop(rect_to_box(rect)).convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return None

    pixels = arr.reshape(-1, 3)
    lum = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    chroma = pixels.max(axis=1) - pixels.min(axis=1)

    bg_cut = float(np.percentile(lum, 55))
    bg_mask = lum <= bg_cut
    if int(bg_mask.sum()) < 4:
        bg_mask = lum <= float(np.percentile(lum, 65))
    if int(bg_mask.sum()) < 4:
        return None

    bg_pixels = pixels[bg_mask]
    background_arr = np.median(bg_pixels, axis=0)
    background = (
        int(round(float(background_arr[0]))),
        int(round(float(background_arr[1]))),
        int(round(float(background_arr[2]))),
    )
    bg_lum = float(np.median(lum[bg_mask]))

    stroke_floor = max(bg_lum + 55.0, float(np.percentile(lum, 75)))
    stroke_mask = (lum >= stroke_floor) & (chroma <= 90.0)
    if int(stroke_mask.sum()) < 4:
        stroke_floor = max(bg_lum + 42.0, float(np.percentile(lum, 70)))
        stroke_mask = (lum >= stroke_floor) & (chroma <= 105.0)
    if int(stroke_mask.sum()) < 4:
        return None

    stroke_lum = lum[stroke_mask]
    core_cut = float(np.percentile(stroke_lum, 55))
    core_mask = stroke_mask & (lum >= core_cut)
    if int(core_mask.sum()) < 4:
        core_mask = stroke_mask
    core_pixels = pixels[core_mask]
    color_arr = np.median(core_pixels, axis=0)
    color = (
        int(round(float(color_arr[0]))),
        int(round(float(color_arr[1]))),
        int(round(float(color_arr[2]))),
    )
    text_lum = max(bg_lum + 1.0, float(np.median(lum[core_mask])))

    alpha_raw = np.clip(((lum - bg_lum) / max(1.0, text_lum - bg_lum)) * 255.0, 0.0, 255.0)
    valid = stroke_mask & (alpha_raw > 8.0)
    alpha_values = alpha_raw[valid].astype(int).tolist()
    if len(alpha_values) < 4:
        return None

    alpha_arr = alpha_raw[valid]
    core_count = int((alpha_arr >= 220.0).sum())
    edge_count = int((alpha_arr < 220.0).sum())
    total = max(1, core_count + edge_count)
    area = max(1, int(rect["width"]) * int(rect["height"]))

    return {
        "color": color,
        "background": background,
        "alpha_values": sorted(alpha_values) or [255],
        "alpha_summary": summarize_values(alpha_values),
        "edge_ratio": edge_count / total,
        "density": len(alpha_values) / area,
    }


def extract_ink_style(image, rect: Dict[str, int]):
    import numpy as np

    background = average_edge_color(image, rect)
    ink_color = extract_ink_color(image, rect)
    bg_luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
    ink_luminance = 0.299 * ink_color[0] + 0.587 * ink_color[1] + 0.114 * ink_color[2]
    luminance_gap = max(1, bg_luminance - ink_luminance)

    arr = np.array(image.crop(rect_to_box(rect)).convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return {"color": ink_color, "background": background, "alpha_values": [255],
                "alpha_summary": summarize_values([]), "edge_ratio": 0.0, "density": 0.0}

    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    is_neutral = (arr.max(axis=2) - arr.min(axis=2)) < 90

    alpha_raw = np.clip(((bg_luminance - luminance) / luminance_gap) * 255, 0, 255)
    valid = is_neutral & (alpha_raw > 0)
    alpha_values = alpha_raw[valid].astype(int).tolist()

    core_count = int((is_neutral & (luminance < 92)).sum())
    edge_count = int((is_neutral & (luminance >= 92) & (luminance < 210)).sum())
    ink_count = core_count + edge_count
    total = max(1, core_count + edge_count)
    area = max(1, rect["width"] * rect["height"])
    return {
        "color": ink_color,
        "background": background,
        "alpha_values": sorted(alpha_values) or [255],
        "alpha_summary": summarize_values(alpha_values),
        "edge_ratio": edge_count / total,
        "density": ink_count / area,
    }


def match_alpha_distribution(mask, target_values):
    import numpy as np
    from PIL import Image

    source_arr = np.array(mask, dtype=np.uint8)
    nonzero = source_arr[source_arr > 0]
    if nonzero.size == 0 or not target_values:
        return mask

    source_sorted = np.sort(nonzero)
    target_sorted = np.array(sorted(target_values), dtype=np.float32)
    source_len = len(source_sorted)
    target_len = len(target_sorted)

    # Build a 256-entry LUT so we never call Python per pixel.
    v = np.arange(1, 256, dtype=np.float32)
    idx = np.searchsorted(source_sorted, v, side="right") - 1
    rank = np.clip(idx, 0, source_len - 1).astype(np.float32)
    percentile = rank / max(1, source_len - 1)
    target_index = np.clip(
        np.round(percentile * (target_len - 1)).astype(int), 0, target_len - 1
    )
    lut = np.zeros(256, dtype=np.uint8)
    lut[1:] = target_sorted[target_index].astype(np.uint8)
    return Image.fromarray(lut[source_arr], mode="L")


def blend_masks(base_mask, matched_mask, strength: float):
    if strength <= 0:
        return base_mask
    if strength >= 1:
        return matched_mask
    import numpy as np
    from PIL import Image

    base_arr = np.array(base_mask, dtype=np.float32)
    matched_arr = np.array(matched_mask, dtype=np.float32)
    blended = np.round(base_arr * (1 - strength) + matched_arr * strength).astype(np.uint8)
    return Image.fromarray(blended, mode="L")


def edge_mask_variants(mask):
    variants = [("base", mask)]
    variants.append(("hard", mask.point(lambda value: 255 if value >= 128 else 0)))
    variants.append(("quantized", mask.point(lambda value: int(round(value / 64) * 64))))
    return variants


def draw_text_in_ocr_rect(image, rect: Dict[str, int], text: str, style):
    from PIL import Image, ImageDraw

    font, bbox, weight_fit = fit_font_to_ink(text, rect, style["density"])
    position = (rect["x"] - bbox[0], rect["y"] - bbox[1])
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if "%" in str(text) and RED_NUMBER_BOLD.is_file():
        sz = int(getattr(font, "size", max(8, rect["height"])))
        draw_metric_red_plus_percent(mask_draw, position, str(text), sz, weight_fit, 255)
    else:
        mask_draw.text(position, text, fill=255, font=font)
    if not _metric_text_has_percent(text):
        mask = tune_mask_weight(mask, style["density"])
    mask_variants = edge_mask_variants(mask)
    target_style = {
        "density": style["density"],
        "edge_ratio": style["edge_ratio"],
        "alpha_summary": style["alpha_summary"],
    }
    candidates = []
    for variant_name, variant_mask in mask_variants:
        matched_mask = match_alpha_distribution(variant_mask, style["alpha_values"])
        for strength in (0, 0.15, 0.3, 0.45, 0.6, 0.75):
            candidate_mask = blend_masks(variant_mask, matched_mask, strength)
            candidate_style = mask_style(candidate_mask)
            candidates.append(
                {
                    "variant": variant_name,
                    "strength": strength,
                    "mask": candidate_mask,
                    "style": candidate_style,
                    "score": style_distance(target_style, candidate_style),
                }
            )
    best = min(candidates, key=lambda item: item["score"])
    style["render_style"] = best["style"]
    style["style_score"] = best["score"]
    style["alpha_match_strength"] = best["strength"]
    style["edge_variant"] = best["variant"]
    mask = best["mask"]

    text_layer = Image.new("RGB", image.size, style["color"])
    return Image.composite(text_layer, image, mask)


def patch_rmse(left, right):
    from PIL import ImageChops, ImageStat

    diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
    stat = ImageStat.Stat(diff)
    mse = sum(value**2 for value in stat.rms) / len(stat.rms)
    return math.sqrt(mse)


def text_mask_for_candidate(
    image_size, text: str, font, position, font_path_hint: Optional[str] = None
):
    from PIL import Image, ImageDraw

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    if "%" in str(text) and _use_red_percent_hybrid(text, font=font, font_path_hint=font_path_hint):
        size = int(getattr(font, "size", 22))
        weight = _weight_from_font_path(font_path_hint)
        draw_metric_red_plus_percent(draw, position, str(text), size, weight, 255)
    else:
        draw.text(position, text, fill=255, font=font)
    return mask


def composite_text_mask(image, mask, color):
    from PIL import Image

    text_layer = Image.new("RGB", image.size, color)
    return Image.composite(text_layer, image, mask)


def adjust_mask_alpha(mask, *, gain: float = 1.0, gamma: float = 1.0):
    import numpy as np
    from PIL import Image

    arr = np.array(mask, dtype=np.float32) / 255.0
    if gamma != 1.0:
        arr = np.power(arr, gamma)
    if gain != 1.0:
        arr *= gain
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray(np.round(arr * 255).astype(np.uint8), mode="L")


def _metric_text_has_percent(text: str) -> bool:
    """``%`` combines tight counters + diagonal; horizontal embolden masks merge into a blob."""

    return "%" in str(text)


def candidate_offsets(height: int):
    delta = max(1, round(height * 0.1))
    return [-delta, 0, delta]


def candidate_masks(base_mask, style):
    variants = []
    for weight_name, weight_mask in weight_mask_variants(base_mask):
        for edge_name, edge_mask in edge_mask_variants(weight_mask):
            variants.append((f"{weight_name}:{edge_name}", edge_mask))
    candidates = []
    for variant_name, variant_mask in variants:
        matched_mask = match_alpha_distribution(variant_mask, style["alpha_values"])
        for strength in (0, 0.25, 0.5, 0.75):
            mask = blend_masks(variant_mask, matched_mask, strength)
            candidates.append((variant_name, strength, mask))
    return candidates


def calibrate_text_render(source_image, clean_image, ink_rect: Dict[str, int], original_text: str, style):
    target_patch_rect = expand_rect(
        ink_rect,
        max(4, ink_rect["height"] // 2),
        {"width": source_image.width, "height": source_image.height},
    )
    target_patch = source_image.crop(rect_to_box(target_patch_rect))
    clean_patch = clean_image.crop(rect_to_box(target_patch_rect))
    target_height = max(1, ink_rect["height"])
    best = None
    size_min = max(8, target_height - 4)
    size_max = min(72, target_height + 8)

    for size in range(size_min, size_max + 1):
        for font_path, font in iter_candidate_fonts(size):
            if _use_red_percent_hybrid(original_text, font=font, font_path_hint=str(font_path)):
                stats = rendered_ink_stats_red_percent_split(
                    original_text, size, _weight_from_font_path(str(font_path))
                )
            else:
                stats = rendered_ink_stats(original_text, font)
            bbox = stats["bbox"]
            rendered_height = bbox[3] - bbox[1]
            if rendered_height <= 0 or abs(rendered_height - target_height) > max(3, target_height * 0.18):
                continue
            for dx in candidate_offsets(target_height):
                for dy in candidate_offsets(target_height):
                    full_position = (ink_rect["x"] + dx - bbox[0], ink_rect["y"] + dy - bbox[1])
                    local_position = (
                        full_position[0] - target_patch_rect["x"],
                        full_position[1] - target_patch_rect["y"],
                    )
                    base_mask = text_mask_for_candidate(
                        target_patch.size,
                        original_text,
                        font,
                        local_position,
                        font_path_hint=str(font_path),
                    )
                    for variant_name, strength, mask in candidate_masks(base_mask, style):
                        if _metric_text_has_percent(original_text):
                            prefix = variant_name.split(":")[0]
                            if prefix in ("w1x", "w1xy", "w2"):
                                continue
                        candidate_patch = composite_text_mask(clean_patch, mask, style["color"])
                        rmse = patch_rmse(target_patch, candidate_patch)
                        candidate_style = mask_style(mask)
                        score = rmse + style_distance(
                            {
                                "density": style["density"],
                                "edge_ratio": style["edge_ratio"],
                                "alpha_summary": style["alpha_summary"],
                            },
                            candidate_style,
                        ) * 8
                        if best is None or score < best["score"]:
                            best = {
                                "score": score,
                                "rmse": rmse,
                                "font_size": size,
                                "font_path": font_path,
                                "dx": dx,
                                "dy": dy,
                                "bbox": bbox,
                                "edge_variant": variant_name,
                                "alpha_match_strength": strength,
                                "render_style": candidate_style,
                                "target_patch_rect": target_patch_rect,
                            }
    if best is None:
        _, bbox, _ = fit_font_to_ink(original_text, ink_rect, style["density"])
        best = {
            "score": None,
            "rmse": None,
            "font_size": ink_rect["height"],
            "font_path": "default-bold",
            "dx": 0,
            "dy": 0,
            "bbox": bbox,
            "edge_variant": "base",
            "alpha_match_strength": 0.5,
            "render_style": None,
            "target_patch_rect": target_patch_rect,
        }
    return best


def calibrate_overlay_font_origin(
    source_image,
    clean_image,
    ink_rect: Dict[str, int],
    original_text: str,
    style,
    calibration,
):
    """Fit the original overlay digits and store the PIL draw origin.

    Tiny feed overlay numbers are too sensitive to bbox-top alignment.  Fit the
    source digits once, then reuse the same ``draw.text`` origin for replacement
    digits so glyph-specific bboxes do not move the text up/down.
    """

    fp_cal = calibration.get("font_path", "") or ""
    font_size = int(calibration.get("font_size", 0) or 0)
    if not fp_cal or font_size <= 0 or not original_text:
        return None
    font = load_font_by_path(fp_cal, font_size)
    if font is None:
        font = load_font(font_size, "medium")

    weight_cal = _weight_from_font_path(str(fp_cal) if fp_cal else None)
    if _use_red_percent_hybrid(str(original_text), font=font, font_path_hint=str(fp_cal) if fp_cal else None):
        bbox = rendered_ink_bbox_red_percent_split(str(original_text), font_size, weight_cal)
    else:
        bbox = rendered_ink_bbox(str(original_text), font)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None

    target_patch_rect = expand_rect(
        ink_rect,
        max(5, int(ink_rect["height"]) // 2),
        {"width": source_image.width, "height": source_image.height},
    )
    target_patch = source_image.crop(rect_to_box(target_patch_rect))
    clean_patch = clean_image.crop(rect_to_box(target_patch_rect))

    target_style = {
        "density": style["density"],
        "edge_ratio": style["edge_ratio"],
        "alpha_summary": style["alpha_summary"],
    }
    base_dx = int(calibration.get("dx", 0))
    base_dy = int(calibration.get("dy", 0))
    delta = max(4, int(round(max(1, int(ink_rect["height"])) * 0.45)))

    best = None
    for dx in range(base_dx - delta, base_dx + delta + 1):
        for dy in range(base_dy - delta, base_dy + delta + 1):
            full_position = (
                int(ink_rect["x"]) + dx - bbox[0],
                int(ink_rect["y"]) + dy - bbox[1],
            )
            local_position = (
                full_position[0] - target_patch_rect["x"],
                full_position[1] - target_patch_rect["y"],
            )
            base_mask = text_mask_for_candidate(
                target_patch.size,
                str(original_text),
                font,
                local_position,
                font_path_hint=str(fp_cal) if fp_cal else None,
            )
            for variant_name, strength, mask in candidate_masks(base_mask, style):
                candidate_patch = composite_text_mask(clean_patch, mask, style["color"])
                candidate_style = mask_style(mask)
                score = patch_rmse(target_patch, candidate_patch) + style_distance(
                    target_style,
                    candidate_style,
                ) * 8
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "rmse": patch_rmse(target_patch, candidate_patch),
                        "position": full_position,
                        "dx": dx,
                        "dy": dy,
                        "bbox": bbox,
                        "edge_variant": variant_name,
                        "alpha_match_strength": strength,
                        "render_style": candidate_style,
                    }

    if best is None:
        return None
    return best


def draw_text_with_calibration(image, ink_rect: Dict[str, int], text: str, style, calibration):
    if not calibration:
        return draw_text_in_ocr_rect(image, ink_rect, text, style)
    font = load_font_by_path(calibration.get("font_path", ""), calibration["font_size"])
    if font is None:
        font = load_font(calibration["font_size"], "bold")
    fp_cal = calibration.get("font_path", "") or ""
    weight_cal = _weight_from_font_path(str(fp_cal) if fp_cal else None)
    if _use_red_percent_hybrid(str(text), font=font, font_path_hint=str(fp_cal) if fp_cal else None):
        bbox = rendered_ink_bbox_red_percent_split(str(text), int(calibration["font_size"]), weight_cal)
    else:
        bbox = rendered_ink_bbox(text, font)
    if "overlay_font_origin" in calibration:
        position = tuple(calibration["overlay_font_origin"])
        old_bbox = calibration.get("overlay_origin_bbox")
        if old_bbox is not None and len(old_bbox) >= 2:
            # Keep the replacement's visible left edge in the same slot.  Tiny
            # overlay digits expose one-pixel left-bearing differences clearly,
            # especially when replacing a leading ``1`` with a wider digit.
            position = (position[0] + int(old_bbox[0]) - int(bbox[0]), position[1])
        calibration["overlay_new_font_origin"] = position
    else:
        position = (
            ink_rect["x"] + calibration["dx"] - bbox[0],
            ink_rect["y"] + calibration["dy"] - bbox[1],
        )
    base_mask = text_mask_for_candidate(
        image.size,
        text,
        font,
        position,
        font_path_hint=str(fp_cal) if fp_cal else None,
    )
    if calibration.get("overlay_direct_mask"):
        mask = adjust_mask_alpha(
            base_mask,
            gain=float(calibration.get("overlay_alpha_gain", 1.0)),
            gamma=float(calibration.get("overlay_alpha_gamma", 1.0)),
        )
        return composite_text_mask(image, mask, calibration.get("overlay_color", style["color"]))
    target_style = {
        "density": style["density"],
        "edge_ratio": style["edge_ratio"],
        "alpha_summary": style["alpha_summary"],
    }
    best = None
    forced_variant = calibration.get("force_edge_variant")
    forced_strength = calibration.get("force_alpha_match_strength")
    if _metric_text_has_percent(text):
        # Forced w1x-style masks bridge the two circles of ``%`` into a solid ink blob.
        forced_variant = None
        forced_strength = None
    for variant_name, strength, mask in candidate_masks(base_mask, style):
        if forced_variant and variant_name != forced_variant:
            continue
        if forced_strength is not None and strength != forced_strength:
            continue
        candidate_style = mask_style(mask)
        score = style_distance(target_style, candidate_style)
        if best is None or score < best[0]:
            best = (score, variant_name, strength, mask, candidate_style)
    if best is None and forced_variant:
        for variant_name, strength, mask in candidate_masks(base_mask, style):
            candidate_style = mask_style(mask)
            score = style_distance(target_style, candidate_style)
            if best is None or score < best[0]:
                best = (score, variant_name, strength, mask, candidate_style)
    chosen_mask = best[3] if best else base_mask
    if best:
        calibration["new_edge_variant"] = best[1]
        calibration["new_alpha_match_strength"] = best[2]
        calibration["new_render_style"] = best[4]
        calibration["new_render_score"] = best[0]
    return composite_text_mask(image, chosen_mask, style["color"])


def patch_field(image, field_name: str, text: str, use_ocr: bool):
    field = NORMAL_FIELDS[field_name]
    rect = locate_text_rect(image, field, use_ocr=use_ocr)
    image = inpaint_or_fill(image, rect)
    image = draw_text(image, rect, text, field)
    return image


def rect_center(rect: Dict[str, int]):
    return (rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2)


def is_metric_value(text: str) -> bool:
    cleaned = "".join(ch for ch in str(text) if ch.isdigit() or ch == "." or ch == "%")
    return bool(cleaned) and any(ch.isdigit() for ch in cleaned)


def normalize_metric_text(text: str) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit() or ch == "." or ch == "%")


def metric_text_changed(original_text: str, new_text: str) -> bool:
    return normalize_metric_text(original_text) != normalize_metric_text(new_text)


def is_pure_metric_text(text: str) -> bool:
    stripped = str(text).strip()
    return bool(stripped) and stripped == normalize_metric_text(stripped)


def find_value_below_label(items, label: str):
    label_item = next((item for item in items if label in item["text"]), None)
    if not label_item:
        raise RuntimeError(f"OCR did not find label: {label}")

    label_rect = label_item["rect"]
    label_cx, _ = rect_center(label_rect)
    candidates = []
    for item in items:
        text = item["text"]
        rect = item["rect"]
        if item is label_item or not is_metric_value(text):
            continue
        dy = rect["y"] - (label_rect["y"] + label_rect["height"])
        if dy < 0 or dy > 80:
            continue
        cx, _ = rect_center(rect)
        dx = abs(cx - label_cx)
        if dx > max(80, label_rect["width"] * 1.2):
            continue
        candidates.append((dy + dx * 0.25, item))

    if not candidates:
        raise RuntimeError(f"OCR did not find value below label: {label}")
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def _header_row_date_like_ocr(raw: str) -> bool:
    """顶部「日期」片（如 ``05-07``）与小眼睛/赞/评同一竖带，需排除。"""

    s = str(raw).strip()
    if not s:
        return False
    if any(ch in s for ch in ("-", "/", "—", "－")) and re.search(r"\d", s):
        return True
    if re.search(r"\d{1,2}\s*月\s*\d{1,2}", s):
        return True
    if "：" in s and re.search(r"\d", s) and len(normalize_metric_text(s)) >= 4:
        return True
    return False


def find_header_view_value(items, image_size=None):
    """顶部笔记卡片「眼睛 / 赞 / 评」一行三个数：取 **从左数第一个** OCR 数字框。

    左侧常有日期 ``MM-DD``，先剔除日期再按 ``x`` 排序，避免误选 ``05-07`` 或赞、评列。
    """

    base_w = BASE_SIZE["width"]
    base_h = BASE_SIZE["height"]
    if image_size is not None:
        sx = image_size[0] / base_w
        sy = image_size[1] / base_h
        img_w = image_size[0]
    else:
        sx = sy = 1.0
        img_w = int(BASE_SIZE["width"])

    y_min = int(210 * sy)
    y_max = int(270 * sy)
    y_slack = int((y_max - y_min) * 0.35)

    candidates = []
    for item in items:
        raw = str(item.get("text", ""))
        if _header_row_date_like_ocr(raw):
            continue
        nt = normalize_metric_text(raw)
        if not nt or "%" in nt:
            continue
        rect = item["rect"]
        if rect["y"] < y_min - y_slack or rect["y"] > y_max + y_slack:
            continue
        if rect["x"] < -8 or rect["x"] > img_w + 8:
            continue
        candidates.append(item)

    if not candidates:
        raise RuntimeError("OCR did not find header view count")

    candidates.sort(key=lambda it: it["rect"]["x"])
    return candidates[0]


def extract_metrics_from_items(items, image_size=None):
    labels = {
        "exposure": "曝光数",
        "views": "观看数",
        "click_rate": "封面点击率",
        "interaction_rate": "互动率",
    }
    result = {}
    for key, lbl in labels.items():
        try:
            item = find_value_below_label(items, lbl)
            result[key] = {
                "label": lbl,
                "text": normalize_metric_text(item["text"]),
                "rect": item["rect"],
            }
        except RuntimeError:
            result[key] = {
                "label": lbl,
                "text": "",
                "rect": None,
            }
    try:
        item = find_header_view_value(items, image_size=image_size)
        result["header_views"] = {
            "label": "顶部观看数",
            "text": normalize_metric_text(item["text"]),
            "rect": item["rect"],
        }
    except RuntimeError:
        result["header_views"] = {
            "label": "顶部观看数",
            "text": "",
            "rect": None,
        }
    return result


def _trim_ink_rect_to_metric_suffix(pixels, bounds: Dict[str, int], raw_text: str) -> Dict[str, int]:
    """Trim a horizontal icon prefix (e.g. eye/play glyphs) off ``bounds``.

    OCR sometimes recognises a small icon as part of the number (e.g.
    ``"◎ 36"``).  ``detect_dark_text_bounds`` then returns a rect covering
    *both* the icon and the digits, which would cause the icon to be erased
    during inpaint.

    We split ``bounds`` into horizontal connected-column groups separated by
    blank columns (no text-like pixels).  If OCR includes a non-metric prefix,
    keep the right-side groups that correspond to the normalized metric text.
    This preserves the leading icon while still keeping all digits, e.g.
    ``"◎ 16"`` -> keep both ``1`` and ``6`` rather than only ``6``.
    """
    metric_text = normalize_metric_text(raw_text)
    if not metric_text or is_pure_metric_text(raw_text):
        return bounds

    x0, y0 = bounds["x"], bounds["y"]
    x1 = x0 + bounds["width"]
    y1 = y0 + bounds["height"]

    has_text = []
    for col in range(x0, x1):
        found = False
        for row in range(y0, y1):
            pixel = pixels[row][col]
            if pixel_looks_like_text(pixel):
                found = True
                break
        has_text.append(found)

    groups: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(has_text):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            groups.append((start, idx - 1))
            start = None
    if start is not None:
        groups.append((start, len(has_text) - 1))

    if len(groups) <= 1:
        return bounds

    # Keep the suffix group run that represents the normalized metric text.
    # In OCR output such as "◎ 16", groups are [icon, 1, 6], so len("16")
    # means "keep the last two groups". Clamp to keep at least one group and
    # avoid dropping everything if OCR merged adjacent digits into one group.
    keep_count = max(1, min(len(metric_text), len(groups)))
    suffix_groups = groups[-keep_count:]
    new_x = x0 + suffix_groups[0][0]
    new_w = suffix_groups[-1][1] - suffix_groups[0][0] + 1

    sub_rows = [pixels[row][new_x:new_x + new_w] for row in range(y0, y1)]
    sub_bounds = detect_dark_text_bounds(sub_rows)
    if not sub_bounds:
        return {"x": new_x, "y": y0, "width": new_w, "height": bounds["height"]}
    return {
        "x": new_x + sub_bounds["x"],
        "y": y0 + sub_bounds["y"],
        "width": sub_bounds["width"],
        "height": sub_bounds["height"],
    }


def _percentile_sorted(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * p))
    idx = max(0, min(len(ys) - 1, idx))
    return ys[idx]


def _luminance_rgb(pixel: Tuple[int, ...]) -> float:
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]


def localize_feed_overlay_views_ink(
    image,
    rect: Dict[str, int],
    raw_text: str = "",
) -> Optional[Dict[str, int]]:
    """Locate light-on-gradient overlay digits (feed thumbnail view count).

    ``detect_dark_text_bounds`` / ``pixel_looks_like_text`` target dark body text;
    cover overlays are often bright strokes on a bright strip, so we score columns
    by high-percentile luminance relative to the crop median background.
    """

    source_rect = dict(rect)
    image_size = {"width": image.width, "height": image.height}
    pad_x = max(2, int(rect["width"]) // 4)
    pad_y = max(4, int(rect["height"]) // 2)
    rect = {
        "x": max(0, int(rect["x"]) - pad_x),
        "y": max(0, int(rect["y"]) - pad_y),
        "width": min(image_size["width"] - max(0, int(rect["x"]) - pad_x), int(rect["width"]) + pad_x * 2),
        "height": min(image_size["height"] - max(0, int(rect["y"]) - pad_y), int(rect["height"]) + pad_y * 2),
    }

    pixels = crop_pixels(image, rect)
    if not pixels:
        return None
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    if height < 3 or width < 4:
        return None

    flat_lums = [_luminance_rgb(pixels[y][x]) for y in range(height) for x in range(width)]
    bg = float(statistics.median(flat_lums))

    scores: List[float] = []
    for x in range(width):
        col = [_luminance_rgb(pixels[y][x]) for y in range(height)]
        scores.append(_percentile_sorted(col, 0.9) - bg)

    def runs_for_delta(delta: float) -> List[Tuple[int, int]]:
        cols = [x for x in range(width) if scores[x] >= delta]
        if not cols:
            return []
        seg: List[Tuple[int, int]] = []
        s0, prev = cols[0], cols[0]
        for c in cols[1:]:
            if c - prev > 2:
                seg.append((s0, prev))
                s0 = c
            prev = c
        seg.append((s0, prev))
        return seg

    delta_order = (18.0, 17.0, 16.0, 15.0, 14.0, 12.0)
    runs: List[Tuple[int, int]] = []
    for d in delta_order:
        runs = runs_for_delta(d)
        if runs:
            break
    if not runs:
        return None

    mega_cut = max(14, int(width * 0.72))
    runs = [(a, b) for a, b in runs if (b - a + 1) < mega_cut]

    merged: List[List[int]] = []
    for a, b in sorted(runs):
        if not merged:
            merged.append([a, b])
            continue
        la, lb = merged[-1]
        if a - lb <= 5:
            merged[-1][1] = max(lb, b)
        else:
            merged.append([a, b])
    runs_merged: List[Tuple[int, int]] = [(m[0], m[1]) for m in merged]

    norm = normalize_metric_text(raw_text or "")
    est_chars = max(1, len(norm))
    min_w = max(6, est_chars * 5)
    max_w_metric = est_chars * 26 + 16
    max_w_crop = min(width - 1, max(int(width * 0.50), est_chars * 22 + 8))
    max_w = min(max_w_metric, max_w_crop)
    max_run_w = min(width - 1, max(int(width * 0.42), est_chars * 22 + 12))
    left_skip = max(8, int(width * 0.13))

    runs_merged = [(a, b) for a, b in runs_merged if (b - a + 1) <= max_run_w]

    def mean_score(a: int, b: int) -> float:
        return sum(scores[x] for x in range(a, b + 1)) / (b - a + 1)

    candidates: List[Tuple[int, int]] = [
        (a, b) for a, b in runs_merged if min_w <= (b - a + 1) <= max_w
    ]
    if not candidates:
        candidates = [(a, b) for a, b in runs_merged if (b - a + 1) >= min_w]

    if not candidates:
        return None

    band_lo = max(min_w - 4, est_chars * 6)
    band_hi = min(max_w + 8, max_run_w)

    def rank_key(ab: Tuple[int, int]) -> Tuple:
        a, b = ab
        ww = b - a + 1
        in_band = band_lo <= ww <= band_hi
        ms = mean_score(a, b)
        past_icon = 1 if a >= left_skip else 0
        # Prefer region right of eye icon, plausible digit width, then contrast score.
        return (past_icon, in_band, ms, a)

    best_a, best_b = max(candidates, key=rank_key)

    bright_thr = min(252.0, bg + 28.0)
    min_y, max_y = height, -1
    for y in range(height):
        xs = pixels[y][best_a : best_b + 1]
        bright = sum(1 for p in xs if _luminance_rgb(p) >= bright_thr)
        if bright >= max(2, len(xs) // 8):
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    if max_y < min_y:
        min_y, max_y = 0, height - 1

    out = {
        "x": rect["x"] + best_a,
        "y": rect["y"] + min_y,
        "width": best_b - best_a + 1,
        "height": max_y - min_y + 1,
    }
    source_h = max(1, int(source_rect["height"]))
    if out["height"] > source_h + max(2, source_h // 4):
        max_up = max(4, int(round(source_h * 0.35)))
        out["y"] = max(0, max(out["y"], int(source_rect["y"]) - max_up))
        out["height"] = min(out["height"], source_h + 2)
    return out


def get_ink_rect(image, rect: Dict[str, int], raw_text: str = "") -> Dict[str, int]:
    pixels = crop_pixels(image, rect)
    bounds = detect_dark_text_bounds(pixels)
    if not bounds:
        return rect
    if raw_text:
        bounds = _trim_ink_rect_to_metric_suffix(pixels, bounds, raw_text)
    return translate_rect(bounds, rect)


def patch_ocr_rect(image, source_image, rect: Dict[str, int], text: str):
    ink_rect = get_ink_rect(source_image, rect)
    image_size = {"width": image.width, "height": image.height}
    # Inpaint only the actual ink pixels so decorative icons (e.g. eye/play)
    # that share the OCR bounding box are preserved.
    padded = expand_rect(ink_rect, max(2, ink_rect["height"] // 5), image_size)
    style = extract_ink_style(source_image, ink_rect)
    image = inpaint_or_fill(image, padded)
    image = draw_text_in_ocr_rect(image, ink_rect, text, style)
    return image, {
        "target": {
            "density": style["density"],
            "edge_ratio": style["edge_ratio"],
            "alpha_summary": style["alpha_summary"],
        },
        "render": style.get("render_style"),
        "score": style.get("style_score"),
        "alpha_match_strength": style.get("alpha_match_strength"),
        "edge_variant": style.get("edge_variant"),
        "color": style["color"],
    }


def column_has_ink(pixels, x: int) -> bool:
    for row in pixels:
        if pixel_looks_like_text(row[x]):
            return True
    return False


def split_ink_columns(image, rect: Dict[str, int]):
    pixels = crop_pixels(image, rect)
    if not pixels:
        return []
    width = len(pixels[0])
    ink_columns = [x for x in range(width) if column_has_ink(pixels, x)]
    if not ink_columns:
        return []

    groups = []
    start = ink_columns[0]
    previous = ink_columns[0]
    for x in ink_columns[1:]:
        if x - previous > 2:
            groups.append((start, previous))
            start = x
        previous = x
    groups.append((start, previous))
    return groups


def segment_glyph_boxes(pixels, gap_threshold: int = 1):
    """Segment a 2D RGB pixel grid into per-glyph bounding boxes.

    Splits along columns where no row contains a text-like pixel. Each emitted
    box is tightened vertically to the actual ink rows so the resulting glyphs
    can be reused without surrounding whitespace.
    """

    if not pixels:
        return []
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    if width == 0:
        return []

    ink_columns = [x for x in range(width) if column_has_ink(pixels, x)]
    if not ink_columns:
        return []

    column_groups = []
    start = ink_columns[0]
    previous = ink_columns[0]
    for x in ink_columns[1:]:
        if x - previous > gap_threshold:
            column_groups.append((start, previous))
            start = x
        previous = x
    column_groups.append((start, previous))

    boxes = []
    for col_start, col_end in column_groups:
        min_y = height
        max_y = -1
        for y in range(height):
            row = pixels[y]
            for x in range(col_start, col_end + 1):
                if pixel_looks_like_text(row[x]):
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    break
        if max_y < min_y:
            continue
        boxes.append(
            {
                "x": col_start,
                "y": min_y,
                "width": col_end - col_start + 1,
                "height": max_y - min_y + 1,
            }
        )
    return boxes


def collect_body_font_source_stats(image, body_targets) -> Dict[str, object]:
    heights = []
    densities = []
    edge_ratios = []
    char_widths = {}
    for _label, _replacement, item in body_targets:
        text = normalize_metric_text(item["text"])
        if not text:
            continue
        ink_rect = get_ink_rect(image, item["rect"], raw_text=item["text"])
        heights.append(ink_rect["height"])
        style = extract_ink_style(image, ink_rect)
        densities.append(style["density"])
        edge_ratios.append(style["edge_ratio"])
        boxes = segment_glyph_boxes(crop_pixels(image, ink_rect))
        if len(boxes) != len(text):
            continue
        for char, box in zip(text, boxes):
            char_widths.setdefault(char, []).append(box["width"])

    target_height = round(_median(heights)) if heights else 1
    return {
        "target_height": target_height,
        "target_density": _median(densities),
        "target_edge_ratio": _median(edge_ratios),
        "char_widths": char_widths,
    }


def proportional_char_boxes(rect: Dict[str, int], text: str):
    if not text:
        return []
    char_width = rect["width"] / len(text)
    return [
        {
            "x": round(rect["x"] + index * char_width),
            "y": rect["y"],
            "width": max(1, round(char_width)),
            "height": rect["height"],
        }
        for index, _ in enumerate(text)
    ]


def segment_char_boxes(image, rect: Dict[str, int], text: str):
    groups = split_ink_columns(image, rect)
    if len(groups) == len(text):
        return [
            {
                "x": rect["x"] + start,
                "y": rect["y"],
                "width": end - start + 1,
                "height": rect["height"],
            }
            for start, end in groups
        ]
    return proportional_char_boxes(rect, text)


def glyph_from_rect(image, rect: Dict[str, int]):
    from PIL import Image
    import numpy as np

    crop = image.crop(rect_to_box(rect)).convert("RGB")
    background = average_edge_color(image, rect)
    ink_color = extract_ink_color(image, rect)
    bg_luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
    ink_luminance = 0.299 * ink_color[0] + 0.587 * ink_color[1] + 0.114 * ink_color[2]
    luminance_gap = max(1, bg_luminance - ink_luminance)

    arr = np.array(crop, dtype=np.float32)  # (H, W, 3)
    is_neutral = (arr.max(axis=2) - arr.min(axis=2)) < 100
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    alpha_raw = np.clip(((bg_luminance - luminance) / luminance_gap) * 255, 0, 255)
    alpha_arr = np.where(is_neutral & (alpha_raw > 0), alpha_raw, 0).astype(np.uint8)

    alpha = Image.fromarray(alpha_arr, mode="L")
    alpha_values = alpha_arr[alpha_arr > 0].tolist()

    bbox = alpha.getbbox()
    if bbox is None:
        return None

    glyph = crop.crop(bbox).convert("RGBA")
    glyph.putalpha(alpha.crop(bbox))
    return {
        "image": glyph,
        "height": glyph.height,
        "width": glyph.width,
        "alpha_values": sorted(alpha_values) or [255],
        "cell_height": crop.height,
        "y_offset": bbox[1],
        "x_offset": bbox[0],
    }


def build_glyph_atlas(image, items):
    atlas = {}
    for item in items:
        if not is_pure_metric_text(item["text"]):
            continue
        text = normalize_metric_text(item["text"])
        if not text:
            continue
        rect = get_ink_rect(image, item["rect"])
        boxes = segment_char_boxes(image, rect, text)
        if len(boxes) != len(text):
            continue
        for char, box in zip(text, boxes):
            glyph = glyph_from_rect(image, box)
            if not glyph:
                continue
            current = atlas.get(char)
            # Prefer larger glyphs; they scale down better than tiny header glyphs.
            if current is None or glyph["height"] > current["height"]:
                atlas[char] = glyph
    return atlas


def build_row_atlas(image, items, y_anchor: int, y_tolerance: int = 30):
    """Build a glyph atlas from OCR items in a narrow vertical band.

    All chosen items must contain only metric characters (digits/./%). The
    returned atlas captures the median glyph height and spacing observed in the
    band so that a new number can be composed at the exact same scale and
    kerning as the source row.
    """

    glyphs = {}
    heights = []
    spacings = []

    for item in items:
        if not is_pure_metric_text(item["text"]):
            continue
        text = normalize_metric_text(item["text"])
        if not text:
            continue
        rect = item["rect"]
        if abs(rect["y"] - y_anchor) > y_tolerance:
            continue

        ink_rect = get_ink_rect(image, rect)
        local_pixels = crop_pixels(image, ink_rect)
        local_boxes = segment_glyph_boxes(local_pixels)
        if len(local_boxes) != len(text):
            continue

        row_height = max(1, ink_rect["height"])
        for index, (char, box) in enumerate(zip(text, local_boxes)):
            global_box = {
                "x": ink_rect["x"] + box["x"],
                "y": ink_rect["y"] + box["y"],
                "width": box["width"],
                "height": box["height"],
            }
            glyph = glyph_from_rect(image, global_box)
            if not glyph:
                continue
            glyph["row_height"] = row_height
            glyph["row_y_offset"] = box["y"]
            heights.append(glyph["height"])
            existing = glyphs.get(char)
            if existing is None or glyph["height"] > existing["height"]:
                glyphs[char] = glyph
            if index < len(local_boxes) - 1:
                next_box = local_boxes[index + 1]
                gap = next_box["x"] - (box["x"] + box["width"])
                spacings.append(gap)

    if not glyphs:
        return None

    sorted_heights = sorted(heights)
    sorted_spacings = sorted(spacings) if spacings else [1]
    return {
        "glyphs": glyphs,
        "reference_height": sorted_heights[len(sorted_heights) // 2],
        "glyph_spacing": sorted_spacings[len(sorted_spacings) // 2],
    }


def compose_text_from_row_atlas(image, atlas, ink_rect: Dict[str, int], text: str):
    """Paste glyph pixels from atlas to recreate `text` at `ink_rect`.

    Returns the new image when every character is available in the atlas, or
    None when at least one character is missing.
    """

    if atlas is None:
        return None

    glyphs = []
    for char in text:
        glyph = atlas["glyphs"].get(char)
        if glyph is None:
            glyph = synthesize_metric_glyph_for_atlas(char, atlas)
            if glyph is None:
                return None
            atlas["glyphs"][char] = glyph
        glyphs.append(glyph)

    from PIL import Image

    target_height = max(1, ink_rect["height"])
    reference_height = max(1, atlas["reference_height"])
    scale = target_height / reference_height

    rendered = []
    for glyph in glyphs:
        source = glyph["image"]
        new_height = max(1, round(source.height * scale))
        new_width = max(1, round(source.width * scale))
        if (new_width, new_height) == source.size:
            scaled_image = source
        else:
            scaled_image = source.resize((new_width, new_height), Image.Resampling.LANCZOS)
        row_height = max(1, glyph.get("row_height", source.height))
        # row_y_offset is the position of this glyph within its source row's ink rect.
        # Scaling preserves the proportion so the dot stays low even when row sizes differ.
        y_offset_in_row = glyph.get("row_y_offset", 0) * (target_height / row_height)
        rendered.append({"image": scaled_image, "y_offset": y_offset_in_row})

    spacing = max(0, round(atlas["glyph_spacing"] * scale))
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    x = ink_rect["x"]
    y_top = ink_rect["y"]
    for entry in rendered:
        glyph_image = entry["image"]
        glyph_y = round(y_top + entry["y_offset"])
        layer.alpha_composite(glyph_image, (round(x), glyph_y))
        x += glyph_image.width + spacing

    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def row_atlas_supports_texts(atlas, texts) -> bool:
    if atlas is None:
        return False
    glyphs = atlas.get("glyphs", {})
    return all(char in glyphs for text in texts for char in str(text))


def should_use_body_row_atlas(atlas, replacement_texts) -> bool:
    """Use source glyphs only when the row atlas fully covers replacements."""
    return row_atlas_supports_texts(atlas, replacement_texts)


def atlas_ink_color(atlas) -> tuple[int, int, int]:
    import numpy as np

    glyphs = atlas.get("glyphs", {}) if atlas else {}
    samples = []
    for glyph in glyphs.values():
        image = glyph.get("image")
        if image is None:
            continue
        arr = np.array(image.convert("RGBA"))
        alpha = arr[:, :, 3] > 0
        if alpha.any():
            samples.append(arr[:, :, :3][alpha])
    if not samples:
        return (0, 0, 0)
    pixels = np.concatenate(samples, axis=0)
    mean = pixels.mean(axis=0)
    return (round(float(mean[0])), round(float(mean[1])), round(float(mean[2])))


def synthesize_metric_glyph_for_atlas(char: str, atlas):
    from PIL import Image, ImageDraw

    if not char:
        return None
    target_height = max(1, int(atlas.get("reference_height", 1)))
    digit_path = (
        str(RED_NUMBER_BOLD)
        if RED_NUMBER_BOLD.is_file()
        else "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    )
    font_size = choose_font_size_for_rendered_height(digit_path, ["0", "8"], target_height)
    if char == "%":
        font = load_percent_glyph_font(font_size, "bold")
    else:
        font = load_font_by_path(digit_path, font_size)
        if font is None:
            font = load_font(font_size, "bold")

    bbox = rendered_ink_bbox(char, font)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    pad = max(2, round(target_height * 0.15))
    mask = Image.new("L", (width + pad * 2, height + pad * 2), 0)
    draw = ImageDraw.Draw(mask)
    draw.text((pad - bbox[0], pad - bbox[1]), char, font=font, fill=255)
    tight = mask.getbbox()
    if tight is None:
        return None

    mask = mask.crop(tight)
    color = atlas_ink_color(atlas)
    glyph = Image.new("RGBA", mask.size, (*color, 255))
    glyph.putalpha(mask)
    return {
        "image": glyph,
        "height": glyph.height,
        "width": glyph.width,
        "row_height": target_height,
        "row_y_offset": max(0, target_height - glyph.height),
        "synthetic": True,
    }


def render_text_from_glyphs(image, rect: Dict[str, int], text: str, atlas) -> bool:
    from PIL import Image

    glyphs = [atlas.get(char) for char in text]
    if any(glyph is None for glyph in glyphs):
        return False

    target_height = max(1, rect["height"])
    scaled = []
    for glyph in glyphs:
        source = glyph["image"]
        scale = target_height / max(1, source.height)
        target_width = max(1, round(source.width * scale))
        scaled.append(source.resize((target_width, target_height), Image.Resampling.LANCZOS))

    spacing = max(1, round(target_height * 0.08))
    total_width = sum(glyph.width for glyph in scaled) + spacing * (len(scaled) - 1)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    x = rect["x"]
    y = rect["y"]
    for glyph in scaled:
        layer.alpha_composite(glyph, (round(x), round(y)))
        x += glyph.width + spacing

    image.paste(Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB"))
    return True


def patch_ocr_rect_with_glyphs(
    image,
    source_image,
    rect: Dict[str, int],
    text: str,
    atlas,
    original_text: str,
    row_atlas=None,
    raw_text: str = "",
    forced_font=None,
    overlay_views_ink: bool = False,
    overlay_thumb: Optional[Dict[str, int]] = None,
    overlay_strip: Optional[Dict[str, int]] = None,
    overlay_anchor_center_y: Optional[float] = None,
    overlay_left_nudge_px: Optional[int] = None,
    overlay_use_input_rect: bool = False,
    overlay_erase_rect: Optional[Dict[str, int]] = None,
):
    ink_rect = None
    if overlay_views_ink and overlay_use_input_rect:
        ink_rect = dict(rect)
    elif overlay_views_ink:
        ink_rect = localize_feed_overlay_views_ink(source_image, rect, raw_text=raw_text)
    if ink_rect is None:
        ink_rect = get_ink_rect(source_image, rect, raw_text=raw_text)

    # Localized bright strokes can be a touch shorter than the source glyphs, but
    # using the whole OCR strip makes feed overlay numbers visibly oversized.
    if overlay_views_ink:
        oh = max(8, int(rect["height"]))
        ih = max(8, int(ink_rect["height"]))
        span = oh - ih
        if span >= 2:
            target_h = min(oh, ih + min(span, max(1, int(round(ih * 0.12)))))
            mid_y = float(ink_rect["y"]) + ih / 2.0
            ink_rect = dict(ink_rect)
            ink_rect["height"] = target_h
            ink_rect["y"] = int(round(mid_y - target_h / 2.0))
            ink_rect["y"] = max(0, ink_rect["y"])

    image_size = {"width": image.width, "height": image.height}

    force_red_metrics = prefer_red_number_metric_render()

    effective_forced_font = forced_font
    if effective_forced_font is None and force_red_metrics and row_atlas is None:
        auto_ff = red_number_forced_font_for_standalone_patch(
            ink_rect=ink_rect,
            original_text=original_text,
            new_text=str(text),
            overlay_views_ink=overlay_views_ink,
        )
        if auto_ff is not None:
            effective_forced_font = auto_ff

    # Erase old glyphs: overlay uses minimal padding and only removes bright strokes.
    if overlay_views_ink:
        erase_basis = dict(overlay_erase_rect) if overlay_erase_rect else dict(rect)
        if ink_rect is not None and not overlay_erase_rect:
            erase_basis["x"] = max(int(rect["x"]), int(ink_rect["x"]) - 1)
            erase_basis["y"] = max(0, min(int(rect["y"]), int(ink_rect["y"])))
            erase_basis["width"] = min(
                image_size["width"] - erase_basis["x"],
                max(int(rect["width"]) + 2, int(ink_rect["width"])),
            )
            erase_basis["height"] = min(
                image_size["height"] - erase_basis["y"],
                max(int(rect["height"]) + 2, int(ink_rect["height"])),
            )
        erase_pad = 2
        if overlay_strip is not None and int(erase_basis.get("x", 0)) >= int(overlay_strip.get("x", 0)) + 28:
            erase_pad = 3
        elif int(erase_basis.get("width", 0)) >= 14:
            erase_pad = 3
        padded = expand_rect(erase_basis, erase_pad, image_size)
        image = inpaint_overlay_views_stroke_fill(image, source_image, padded)
    else:
        padded = expand_rect(ink_rect, max(2, ink_rect["height"] // 5), image_size)
        image = inpaint_or_fill(image, padded)

    if row_atlas is not None and not force_red_metrics:
        glyph_ink = ink_rect
        if overlay_views_ink:
            probe_font = load_font(max(8, min(72, ink_rect["height"])), "bold")
            nx = _overlay_views_left_nudge_px(original_text, text, font=probe_font) + int(overlay_left_nudge_px or 0)
            if nx:
                glyph_ink = dict(ink_rect)
                glyph_ink["x"] = ink_rect["x"] + nx
        composed = compose_text_from_row_atlas(image, row_atlas, glyph_ink, text)
        if composed is not None:
            return composed, {
                "mode": "row_atlas",
                "reference_height": row_atlas["reference_height"],
                "glyph_spacing": row_atlas["glyph_spacing"],
                "ink_rect": glyph_ink,
            }

    glyph_rect = ink_rect
    if overlay_views_ink:
        probe_font = load_font(max(8, min(72, ink_rect["height"])), "bold")
        nx = _overlay_views_left_nudge_px(original_text, text, font=probe_font) + int(overlay_left_nudge_px or 0)
        if nx:
            glyph_rect = dict(ink_rect)
            glyph_rect["x"] = ink_rect["x"] + nx
    if not force_red_metrics and render_text_from_glyphs(image, glyph_rect, text, atlas):
        return image, {"mode": "glyph"}
    if overlay_views_ink:
        style = extract_light_overlay_ink_style(source_image, ink_rect) or extract_ink_style(source_image, ink_rect)
        lite = extract_light_overlay_text_color(source_image, ink_rect)
        if lite:
            style["color"] = lite
    else:
        style = extract_ink_style(source_image, ink_rect)
    calibration = calibrate_text_render(source_image, image, ink_rect, original_text, style)
    if effective_forced_font:
        calibration["font_size"] = effective_forced_font["font_size"]
        calibration["font_path"] = effective_forced_font["font_path"]
        calibration["forced_font"] = True
        if "force_edge_variant" in effective_forced_font:
            calibration["force_edge_variant"] = effective_forced_font["force_edge_variant"]
        if "force_alpha_match_strength" in effective_forced_font:
            calibration["force_alpha_match_strength"] = effective_forced_font["force_alpha_match_strength"]
        if "font_match" in effective_forced_font:
            calibration["font_match"] = effective_forced_font["font_match"]
        if overlay_views_ink:
            if effective_forced_font.get("overlay_direct_mask"):
                calibration["overlay_direct_mask"] = True
            calibration["dx"] = int(effective_forced_font.get("overlay_visual_dx", calibration.get("dx", 0)))
            calibration["dy"] = int(effective_forced_font.get("overlay_visual_dy", calibration.get("dy", 0)))
            calibration["overlay_alpha_gain"] = effective_forced_font.get("overlay_alpha_gain", 1.0)
            calibration["overlay_alpha_gamma"] = effective_forced_font.get("overlay_alpha_gamma", 1.0)
            calibration["overlay_color"] = effective_forced_font.get("overlay_color", style.get("color", (255, 255, 255)))
    if overlay_views_ink:
        origin_fit = calibrate_overlay_font_origin(
            source_image,
            image,
            ink_rect,
            original_text,
            style,
            calibration,
        )
        if origin_fit is not None:
            calibration["overlay_font_origin"] = origin_fit["position"]
            calibration["overlay_origin_source_text"] = str(original_text)
            calibration["overlay_origin_dx"] = origin_fit["dx"]
            calibration["overlay_origin_dy"] = origin_fit["dy"]
            calibration["overlay_origin_bbox"] = origin_fit["bbox"]
            calibration["overlay_origin_score"] = origin_fit["score"]
            calibration["overlay_origin_rmse"] = origin_fit["rmse"]
            calibration["overlay_origin_edge_variant"] = origin_fit["edge_variant"]
            calibration["overlay_origin_alpha_match_strength"] = origin_fit["alpha_match_strength"]
        cal_font = load_font_by_path(calibration.get("font_path", ""), calibration["font_size"])
        if cal_font is None:
            cal_font = load_font(calibration["font_size"], "bold")
        if "overlay_font_origin" not in calibration:
            nx = _overlay_views_left_nudge_px(original_text, text, font=cal_font) + int(overlay_left_nudge_px or 0)
            calibration["dx"] = int(calibration.get("dx", 0)) + nx
            if overlay_left_nudge_px:
                calibration["overlay_left_nudge_px"] = int(overlay_left_nudge_px)
        if overlay_anchor_center_y is not None and "overlay_font_origin" not in calibration:
            fp_cal = calibration.get("font_path", "") or ""
            weight_cal = _weight_from_font_path(str(fp_cal) if fp_cal else None)
            if _use_red_percent_hybrid(str(text), font=cal_font, font_path_hint=str(fp_cal) if fp_cal else None):
                bbox = rendered_ink_bbox_red_percent_split(str(text), int(calibration["font_size"]), weight_cal)
            else:
                bbox = rendered_ink_bbox(str(text), cal_font)
            rendered_h = max(1, bbox[3] - bbox[1])
            current_center_y = float(ink_rect["y"]) + int(calibration.get("dy", 0)) + rendered_h / 2.0
            center_delta = int(round(float(overlay_anchor_center_y) - current_center_y))
            center_delta = max(-4, min(4, center_delta))
            calibration["dy"] = int(calibration.get("dy", 0)) + center_delta
            calibration["overlay_anchor_center_y"] = float(overlay_anchor_center_y)
            calibration["overlay_anchor_dy_delta"] = center_delta
    image = draw_text_with_calibration(image, ink_rect, text, style, calibration)
    return image, {
        "mode": "font",
        "calibration": calibration,
        "target": {
            "density": style["density"],
            "edge_ratio": style["edge_ratio"],
            "alpha_summary": style["alpha_summary"],
        },
        "render": calibration.get("render_style") if calibration else None,
        "score": calibration.get("score") if calibration else None,
        "rmse": calibration.get("rmse") if calibration else None,
        "alpha_match_strength": calibration.get("alpha_match_strength") if calibration else None,
        "edge_variant": calibration.get("edge_variant") if calibration else None,
        "color": style["color"],
    }


def refine_header_number_rect(image, rect: Dict[str, int]) -> Dict[str, int]:
    pixels = crop_pixels(image, rect)
    if not pixels:
        return rect

    height = len(pixels)
    width = len(pixels[0]) if height else 0
    columns = []
    for x in range(width):
        count = 0
        for y in range(height):
            if pixel_looks_like_text(pixels[y][x]):
                count += 1
        if count >= 2:
            columns.append(x)

    if not columns:
        return rect

    groups = []
    start = columns[0]
    previous = columns[0]
    for x in columns[1:]:
        if x - previous > 2:
            groups.append((start, previous))
            start = x
        previous = x
    groups.append((start, previous))

    if len(groups) > 1:
        split_index = 0
        split_gap = -1
        for index in range(len(groups) - 1):
            gap = groups[index + 1][0] - groups[index][1]
            if gap > split_gap:
                split_gap = gap
                split_index = index + 1
        number_groups = groups[split_index:]
        min_x = number_groups[0][0]
        max_x = number_groups[-1][1]
    else:
        min_x, max_x = groups[0]
    min_y = height
    max_y = -1
    for y in range(height):
        for x in range(min_x, max_x + 1):
            if pixel_looks_like_text(pixels[y][x]):
                min_y = min(min_y, y)
                max_y = max(max_y, y)

    if max_y < min_y:
        return rect

    return {
        "x": rect["x"] + min_x,
        "y": rect["y"] + min_y,
        "width": max(1, max_x - min_x + 1),
        "height": max(1, max_y - min_y + 1),
    }


def beautify_normal_with_ocr(args, metrics, on_progress=None):
    from PIL import Image
    import numpy as np

    def _prog(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    _prog("加载图片…")
    image = Image.open(args.normal).convert("RGB")
    source_image = image.copy()

    _prog("OCR 识别文字…")
    items = detect_items_with_paddle(np.array(image))
    fallback_atlas = build_glyph_atlas(source_image, items) if args.glyph_atlas else {}

    body_targets = []
    for label_key, text in (
        ("曝光数", metrics["exposure_text"]),
        ("观看数", metrics["views_text"]),
        ("封面点击率", metrics["click_rate_text"]),
        ("互动率", metrics["interaction_rate_text"]),
    ):
        item = find_value_below_label(items, label_key)
        body_targets.append((label_key, text, item))

    body_y_anchor = (
        sum(target[2]["rect"]["y"] for target in body_targets) // len(body_targets)
        if body_targets
        else 0
    )
    if prefer_red_number_metric_render():
        body_atlas = None
    else:
        body_atlas = build_row_atlas(source_image, items, body_y_anchor, y_tolerance=300)
        if not should_use_body_row_atlas(body_atlas, [target[1] for target in body_targets]):
            body_atlas = None
    body_forced_font = None
    if body_targets and body_atlas is None:
        source_stats = collect_body_font_source_stats(source_image, body_targets)
        body_font = {
            "font_path": BODY_NATIVE_FONT_PATH,
            "font_size": choose_font_size_for_rendered_height(
                BODY_NATIVE_FONT_PATH,
                [target[1] for target in body_targets],
                int(source_stats["target_height"]),
            ),
            "score": None,
            "matched_chars": len(source_stats.get("char_widths", {})),
            "target_height": source_stats["target_height"],
        }
        body_forced_font = {
            "font_size": max(8, int(body_font["font_size"]) + BODY_NATIVE_FONT_SIZE_ADJUST),
            "font_path": str(body_font["font_path"]),
            "font_match": body_font,
        }
        if BODY_NATIVE_FORCE_EDGE_VARIANT is not None:
            body_forced_font["force_edge_variant"] = BODY_NATIVE_FORCE_EDGE_VARIANT
        if BODY_NATIVE_FORCE_ALPHA_STRENGTH is not None:
            body_forced_font["force_alpha_match_strength"] = BODY_NATIVE_FORCE_ALPHA_STRENGTH

    try:
        header_raw = find_header_view_value(items, image_size=image.size)
        full_rect = dict(header_raw["rect"])
        ink_probe = get_ink_rect(source_image, full_rect, raw_text=str(header_raw["text"]))
        refined_rect = refine_header_number_rect(source_image, full_rect)
        header_item = dict(header_raw)
        header_item["rect"] = refined_rect
        header_forced_font = build_header_views_forced_font(
            ink_rect=ink_probe,
            original_text=normalize_metric_text(header_raw["text"]),
            new_text=str(metrics["header_views_text"]),
        )
        header_patches = [
            ("header_views", metrics["header_views_text"], header_item, None, header_forced_font),
        ]
    except RuntimeError:
        print("[warn] header view count not found, skipping header patch")
        header_patches = []

    patches = header_patches[:]
    for label_key, text, item in body_targets:
        patches.append((label_key, text, item, body_atlas, body_forced_font))

    total = len(patches)
    for i, patch in enumerate(patches):
        if len(patch) == 4:
            label, text, item, row_atlas = patch
            forced_font = None
        else:
            label, text, item, row_atlas, forced_font = patch
        _prog(f"渲染数字 {i + 1}/{total}…")
        rect = item["rect"]
        original_text = normalize_metric_text(item["text"])
        if label != "header_views" and not metric_text_changed(original_text, str(text)):
            if args.style_report:
                print(json.dumps({str(label): {"mode": "unchanged", "text": original_text}}, ensure_ascii=False))
            continue
        image, report = patch_ocr_rect_with_glyphs(
            image,
            source_image,
            rect,
            str(text),
            fallback_atlas,
            original_text,
            row_atlas,
            raw_text=item["text"],
            forced_font=forced_font,
        )
        if args.style_report:
            print(json.dumps({str(label): report}, ensure_ascii=False))

    _prog("完成")
    return image


def inspect_normal(args):
    from PIL import Image
    import numpy as np

    image = Image.open(args.normal).convert("RGB")
    items = detect_items_with_paddle(np.array(image))
    metrics = extract_metrics_from_items(items, image_size=image.size)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def beautify_normal(args):
    from PIL import Image

    if args.inspect:
        inspect_normal(args)
        return None

    missing = [
        name
        for name in ["output", "exposure", "views"]
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments for generation: {', '.join('--' + name for name in missing)}")

    metrics = calculate_metrics(
        exposure=args.exposure,
        views=args.views,
        likes=args.likes,
        comments=args.comments,
        collects=args.collects,
        shares=args.shares,
    )
    if args.ocr:
        image = beautify_normal_with_ocr(args, metrics)
    else:
        image = Image.open(args.normal).convert("RGB")
        patches = [
            ("exposure", metrics["exposure_text"]),
            ("views", metrics["views_text"]),
            ("click_rate", metrics["click_rate_text"]),
            ("interaction_rate", metrics["interaction_rate_text"]),
        ]
        for field_name, text in patches:
            image = patch_field(image, field_name, str(text), use_ocr=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=88)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="Local beautify-data image prototype")
    parser.add_argument("--normal", required=True, help="Path to normal.jpg-like screenshot")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--views", type=float)
    parser.add_argument("--likes", type=float, default=0)
    parser.add_argument("--comments", type=float, default=0)
    parser.add_argument("--collects", type=float, default=0)
    parser.add_argument("--shares", type=float, default=0)
    parser.add_argument("--ocr", action="store_true", help="Use local PaddleOCR for text box detection")
    parser.add_argument("--inspect", action="store_true", help="Extract current metric values with PaddleOCR")
    parser.add_argument("--style-report", action="store_true", help="Print style matching diagnostics")
    parser.add_argument(
        "--glyph-atlas",
        action="store_true",
        help="Experimental: compose new numbers from source-image glyphs when possible",
    )
    return parser


def main(argv: Optional[list] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = beautify_normal(args)
    if output:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
