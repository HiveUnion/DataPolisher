from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

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

FONT_CANDIDATE_PATHS = [
    "/System/Library/Fonts/KohinoorGujarati.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/SFNSRounded.ttf",
    "/System/Library/Fonts/ADTNumeric.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf",
]


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
    width, height = image.size
    padding = max(3, min(rect["width"], rect["height"]) // 5)
    samples = []
    sample_rects = [
        {"x": rect["x"] - padding, "y": rect["y"], "width": padding, "height": rect["height"]},
        {
            "x": rect["x"] + rect["width"],
            "y": rect["y"],
            "width": padding,
            "height": rect["height"],
        },
        {"x": rect["x"], "y": rect["y"] - padding, "width": rect["width"], "height": padding},
        {
            "x": rect["x"],
            "y": rect["y"] + rect["height"],
            "width": rect["width"],
            "height": padding,
        },
    ]

    for item in sample_rects:
        clipped = {
            "x": max(0, item["x"]),
            "y": max(0, item["y"]),
            "width": max(1, min(width - max(0, item["x"]), item["width"])),
            "height": max(1, min(height - max(0, item["y"]), item["height"])),
        }
        samples.extend(image.crop(rect_to_box(clipped)).convert("RGB").getdata())

    if not samples:
        return (255, 255, 255)

    bright_samples = []
    for pixel in samples:
        r, g, b = pixel
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance > 218 and max(pixel) - min(pixel) < 28:
            bright_samples.append(pixel)
    if bright_samples:
        samples = bright_samples

    total = [0, 0, 0]
    for r, g, b in samples:
        total[0] += r
        total[1] += g
        total[2] += b
    count = len(samples)
    return (round(total[0] / count), round(total[1] / count), round(total[2] / count))


def inpaint_or_fill(image, rect: Dict[str, int]):
    from PIL import ImageDraw

    patched = image.copy()
    draw = ImageDraw.Draw(patched)
    color = average_edge_color(image, rect)
    draw.rectangle(rect_to_box(rect), fill=color)
    return patched


def load_font(size: int, weight: str):
    from PIL import ImageFont

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
    yielded = False
    for path in FONT_CANDIDATE_PATHS:
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
    bbox = mask.getbbox()
    if bbox is None:
        return {
            "density": 0,
            "edge_ratio": 0,
            "alpha_summary": {"p10": 0, "p50": 0, "p90": 0},
        }
    crop = mask.crop(bbox)
    values = [value for value in crop.getdata() if value > 0]
    area = max(1, crop.width * crop.height)
    core_count = sum(1 for value in values if value >= 220)
    edge_count = sum(1 for value in values if 0 < value < 220)
    total = max(1, core_count + edge_count)
    return {
        "density": len(values) / area,
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
    density_score = abs(target["density"] - candidate["density"]) * 8
    edge_score = abs(target["edge_ratio"] - candidate["edge_ratio"]) * 8
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
    target_height = max(1, target_rect["height"])
    best = None

    for size in range(8, 72):
        for weight in ("regular", "medium", "bold"):
            font = load_font(size, weight)
            stats = rendered_ink_stats(text, font)
            bbox = stats["bbox"]
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            height_score = abs(height - target_height) * 2.0
            density_score = abs(stats["density"] - target_density) * 18.0
            width_score = abs(width - target_rect["width"]) * 0.04
            score = height_score + density_score + width_score
            if best is None or score < best[0]:
                best = (score, font, bbox)

    if best is None:
        font = load_font(target_height, weight)
        return font, rendered_ink_bbox(text, font)

    return best[1], best[2]


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
    draw.text(position, text, fill=(34, 34, 34), font=font)
    return image


def extract_ink_color(image, rect: Dict[str, int]):
    pixels = crop_pixels(image, rect)
    samples = []
    for row in pixels:
        for pixel in row:
            r, g, b = pixel
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            if luminance < 180 and max(pixel) - min(pixel) < 90:
                samples.append(pixel)

    if not samples:
        return (34, 34, 34)

    samples.sort(key=lambda item: 0.299 * item[0] + 0.587 * item[1] + 0.114 * item[2])
    core = samples[: max(1, int(len(samples) * 0.65))]
    return tuple(round(sum(pixel[index] for pixel in core) / len(core)) for index in range(3))


def extract_ink_style(image, rect: Dict[str, int]):
    pixels = crop_pixels(image, rect)
    core_count = 0
    edge_count = 0
    ink_count = 0
    background = average_edge_color(image, rect)
    ink_color = extract_ink_color(image, rect)
    bg_luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
    ink_luminance = 0.299 * ink_color[0] + 0.587 * ink_color[1] + 0.114 * ink_color[2]
    luminance_gap = max(1, bg_luminance - ink_luminance)
    alpha_values = []

    for row in pixels:
        for pixel in row:
            r, g, b = pixel
            if max(pixel) - min(pixel) >= 90:
                continue
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            alpha = max(0, min(255, round(((bg_luminance - luminance) / luminance_gap) * 255)))
            if alpha > 0:
                alpha_values.append(alpha)
            if luminance < 92:
                core_count += 1
                ink_count += 1
            elif luminance < 210:
                edge_count += 1
                ink_count += 1

    total = max(1, core_count + edge_count)
    area = max(1, rect["width"] * rect["height"])
    edge_ratio = edge_count / total
    return {
        "color": ink_color,
        "background": background,
        "alpha_values": sorted(alpha_values) or [255],
        "alpha_summary": summarize_values(alpha_values),
        "edge_ratio": edge_ratio,
        "density": ink_count / area,
    }


def match_alpha_distribution(mask, target_values):
    source_values = sorted(value for value in mask.getdata() if value > 0)
    if not source_values or not target_values:
        return mask

    source_len = len(source_values)
    target_len = len(target_values)

    def map_value(value):
        if value <= 0:
            return 0
        lo = 0
        hi = source_len
        while lo < hi:
            mid = (lo + hi) // 2
            if source_values[mid] <= value:
                lo = mid + 1
            else:
                hi = mid
        rank = max(0, lo - 1)
        percentile = rank / max(1, source_len - 1)
        target_index = min(target_len - 1, round(percentile * (target_len - 1)))
        return target_values[target_index]

    return mask.point(map_value)


def blend_masks(base_mask, matched_mask, strength: float):
    if strength <= 0:
        return base_mask
    if strength >= 1:
        return matched_mask

    base_data = list(base_mask.getdata())
    matched_data = list(matched_mask.getdata())
    blended = [
        round(base_value * (1 - strength) + matched_value * strength)
        for base_value, matched_value in zip(base_data, matched_data)
    ]
    result = base_mask.copy()
    result.putdata(blended)
    return result


def edge_mask_variants(mask):
    variants = [("base", mask)]
    variants.append(("hard", mask.point(lambda value: 255 if value >= 128 else 0)))
    variants.append(("quantized", mask.point(lambda value: int(round(value / 64) * 64))))
    return variants


def draw_text_in_ocr_rect(image, rect: Dict[str, int], text: str, style):
    from PIL import Image, ImageDraw

    font, bbox = fit_font_to_ink(text, rect, style["density"])
    position = (rect["x"] - bbox[0], rect["y"] - bbox[1])
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.text(position, text, fill=255, font=font)
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


def text_mask_for_candidate(image_size, text: str, font, position):
    from PIL import Image, ImageDraw

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.text(position, text, fill=255, font=font)
    return mask


def composite_text_mask(image, mask, color):
    from PIL import Image

    text_layer = Image.new("RGB", image.size, color)
    return Image.composite(text_layer, image, mask)


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
                    base_mask = text_mask_for_candidate(target_patch.size, original_text, font, local_position)
                    for variant_name, strength, mask in candidate_masks(base_mask, style):
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
        font, bbox = fit_font_to_ink(original_text, ink_rect, style["density"])
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


def draw_text_with_calibration(image, ink_rect: Dict[str, int], text: str, style, calibration):
    if not calibration:
        return draw_text_in_ocr_rect(image, ink_rect, text, style)
    font = load_font_by_path(calibration.get("font_path", ""), calibration["font_size"])
    if font is None:
        font = load_font(calibration["font_size"], "bold")
    bbox = rendered_ink_bbox(text, font)
    position = (
        ink_rect["x"] + calibration["dx"] - bbox[0],
        ink_rect["y"] + calibration["dy"] - bbox[1],
    )
    base_mask = text_mask_for_candidate(image.size, text, font, position)
    target_style = {
        "density": style["density"],
        "edge_ratio": style["edge_ratio"],
        "alpha_summary": style["alpha_summary"],
    }
    best = None
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


def find_header_view_value(items, image_size=None):
    # Base search window designed for BASE_SIZE (460x997).
    # Scale proportionally when a different resolution image is used.
    base_w = BASE_SIZE["width"]
    base_h = BASE_SIZE["height"]
    if image_size is not None:
        sx = image_size[0] / base_w
        sy = image_size[1] / base_h
    else:
        sx = sy = 1.0

    y_min = int(210 * sy)
    y_max = int(270 * sy)
    x_min = int(100 * sx)
    x_max = int(210 * sx)
    # Allow a 30% slack on each side to absorb minor layout differences.
    y_slack = int((y_max - y_min) * 0.3)
    x_slack = int((x_max - x_min) * 0.3)

    candidates = []
    for item in items:
        text = normalize_metric_text(item["text"])
        rect = item["rect"]
        if not text or "%" in text:
            continue
        if rect["y"] < y_min - y_slack or rect["y"] > y_max + y_slack:
            continue
        if rect["x"] < x_min - x_slack or rect["x"] > x_max + x_slack:
            continue
        candidates.append((rect["x"], item))

    if not candidates:
        raise RuntimeError("OCR did not find header view count")
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def extract_metrics_from_items(items):
    labels = {
        "exposure": "曝光数",
        "views": "观看数",
        "click_rate": "封面点击率",
        "interaction_rate": "互动率",
    }
    result = {}
    for key, label in labels.items():
        try:
            item = find_value_below_label(items, label)
            result[key] = {
                "label": label,
                "text": normalize_metric_text(item["text"]),
                "rect": item["rect"],
            }
        except RuntimeError:
            result[key] = {
                "label": label,
                "text": "",
                "rect": None,
            }
    try:
        item = find_header_view_value(items)
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


def get_ink_rect(image, rect: Dict[str, int]) -> Dict[str, int]:
    bounds = detect_dark_text_bounds(crop_pixels(image, rect))
    if not bounds:
        return rect
    return translate_rect(bounds, rect)


def patch_ocr_rect(image, source_image, rect: Dict[str, int], text: str):
    ink_rect = get_ink_rect(source_image, rect)
    image_size = {"width": image.width, "height": image.height}
    union_x = min(rect["x"], ink_rect["x"])
    union_y = min(rect["y"], ink_rect["y"])
    union_x2 = max(rect["x"] + rect["width"], ink_rect["x"] + ink_rect["width"])
    union_y2 = max(rect["y"] + rect["height"], ink_rect["y"] + ink_rect["height"])
    union_rect = {
        "x": union_x, "y": union_y,
        "width": max(1, union_x2 - union_x),
        "height": max(1, union_y2 - union_y),
    }
    padded = expand_rect(union_rect, max(2, ink_rect["height"] // 5), image_size)
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

    crop = image.crop(rect_to_box(rect)).convert("RGB")
    background = average_edge_color(image, rect)
    ink_color = extract_ink_color(image, rect)
    bg_luminance = 0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2]
    ink_luminance = 0.299 * ink_color[0] + 0.587 * ink_color[1] + 0.114 * ink_color[2]
    luminance_gap = max(1, bg_luminance - ink_luminance)
    alpha = Image.new("L", crop.size, 0)
    alpha_values = []

    for y in range(crop.height):
        for x in range(crop.width):
            r, g, b = crop.getpixel((x, y))
            if max(r, g, b) - min(r, g, b) >= 100:
                continue
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            value = max(0, min(255, round(((bg_luminance - luminance) / luminance_gap) * 255)))
            if value > 0:
                alpha.putpixel((x, y), value)
                alpha_values.append(value)

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
            return None
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
):
    ink_rect = get_ink_rect(source_image, rect)
    image_size = {"width": image.width, "height": image.height}

    # Use the union of the OCR bounding box and the ink rect so that coloured
    # icons (e.g. the eye / play icon next to a number) are also inpainted even
    # though they don't pass the dark-pixel "text" test.
    union_x = min(rect["x"], ink_rect["x"])
    union_y = min(rect["y"], ink_rect["y"])
    union_x2 = max(rect["x"] + rect["width"], ink_rect["x"] + ink_rect["width"])
    union_y2 = max(rect["y"] + rect["height"], ink_rect["y"] + ink_rect["height"])
    union_rect = {
        "x": union_x,
        "y": union_y,
        "width": max(1, union_x2 - union_x),
        "height": max(1, union_y2 - union_y),
    }
    padded = expand_rect(union_rect, max(2, ink_rect["height"] // 5), image_size)
    image = inpaint_or_fill(image, padded)

    if row_atlas is not None:
        composed = compose_text_from_row_atlas(image, row_atlas, ink_rect, text)
        if composed is not None:
            return composed, {
                "mode": "row_atlas",
                "reference_height": row_atlas["reference_height"],
                "glyph_spacing": row_atlas["glyph_spacing"],
                "ink_rect": ink_rect,
            }

    if render_text_from_glyphs(image, ink_rect, text, atlas):
        return image, {"mode": "glyph"}
    style = extract_ink_style(source_image, ink_rect)
    calibration = calibrate_text_render(source_image, image, ink_rect, original_text, style)
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


def beautify_normal_with_ocr(args, metrics):
    from PIL import Image
    import numpy as np

    image = Image.open(args.normal).convert("RGB")
    source_image = image.copy()
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
    body_atlas = build_row_atlas(source_image, items, body_y_anchor, y_tolerance=300)

    try:
        header_item = find_header_view_value(items, image_size=image.size)
        header_atlas = build_row_atlas(
            source_image, items, header_item["rect"]["y"], y_tolerance=20
        )
        header_patches = [("header_views", metrics["views_text"], header_item, header_atlas)]
    except RuntimeError:
        print("[warn] header view count not found, skipping header patch")
        header_patches = []

    patches = header_patches[:]
    for label_key, text, item in body_targets:
        patches.append((label_key, text, item, body_atlas))

    for label, text, item, row_atlas in patches:
        rect = item["rect"]
        original_text = normalize_metric_text(item["text"])
        image, report = patch_ocr_rect_with_glyphs(
            image, source_image, rect, str(text), fallback_atlas, original_text, row_atlas
        )
        if args.style_report:
            print(json.dumps({str(label): report}, ensure_ascii=False))

    return image


def inspect_normal(args):
    from PIL import Image
    import numpy as np

    image = Image.open(args.normal).convert("RGB")
    items = detect_items_with_paddle(np.array(image))
    metrics = extract_metrics_from_items(items)
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

