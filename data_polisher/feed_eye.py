"""按标题（模糊匹配）定位信息流卡片，并替换「小眼睛」浏览数字。

常见 App **笔记列表**：浏览量在 **封面缩略图左下角**（半透明眼睛图标右侧），标题在封面下方。
算法根据标题推断所属栅格列与近似正方形封面区域，再在封面底部左侧条带内挑选 OCR 框并修补。

位数受限后 **不再** 拉长或修补半透明胶囊背景，仅在原数字区域内擦除并重绘笔画。
"""

from __future__ import annotations

import difflib
import random
import re
import unicodedata
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from . import cli


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def _is_title_like(text: str) -> bool:
    t = str(text).strip()
    if len(t) < 2:
        return False
    if cli.is_pure_metric_text(t):
        return False
    if _has_cjk(t):
        return True
    return len(t) >= 8 and bool(re.search(r"[A-Za-z]", t))


def _title_match_score(query: str, candidate: str) -> float:
    q, t = query.strip(), candidate.strip()
    if not q or not t:
        return 0.0
    if q in t:
        return 1.0 + min(len(q), len(t)) / max(len(t), 1) * 0.1
    if t in q:
        return 0.88 + min(len(t), len(q)) / max(len(q), 1) * 0.08
    query_parts = [
        part
        for part in re.split(r"[\s,，;；/|]+", q)
        if len(part) >= 2 and not re.fullmatch(r"\d{1,2}[-/.]\d{1,2}", part)
    ]
    if query_parts:
        hits = [part for part in query_parts if part in t]
        if hits:
            longest = max(len(part) for part in hits)
            return 0.82 + min(0.16, longest / max(len(t), 1) * 0.45)
    return float(difflib.SequenceMatcher(None, q, t).ratio())


def pick_best_title_item(items: list, query: str):
    if not query.strip():
        raise ValueError("标题关键词不能为空")
    best = None
    best_score = -1.0
    for item in items:
        text = str(item.get("text", ""))
        if not _is_title_like(text):
            continue
        s = _title_match_score(query, text)
        if s > best_score:
            best_score = s
            best = item
    if best is None:
        raise RuntimeError("OCR 未识别到可作为标题的文本，请换截图或缩短/改写关键词")
    if best_score < 0.35:
        raise RuntimeError(
            f"标题匹配度过低（{best_score:.2f}），最接近的 OCR 文本是：{best['text']!r}"
        )
    return best, best_score


_VIEWS_PURE_NUMERIC = re.compile(r"^[0-9]+(?:\.[0-9]+)?万?$")


def _is_pure_views_numeric_ocr(raw: str) -> bool:
    t = unicodedata.normalize("NFKC", str(raw)).strip().replace(" ", "")
    if not t or "%" in t:
        return False
    if not _VIEWS_PURE_NUMERIC.fullmatch(t):
        return False
    core = t.replace("万", "").replace(".", "")
    if len(core) > 7:
        return False
    return any(ch.isdigit() for ch in t)


def _mixed_metric_fallback_ok(raw: str) -> bool:
    t = unicodedata.normalize("NFKC", str(raw)).strip().replace(" ", "")
    if not t:
        return False
    digits = sum(1 for c in t if c.isdigit())
    if len(t) <= 8:
        return digits >= 1
    if len(t) <= 14:
        return digits >= 2 and digits >= len(t) * 0.22
    return digits >= 2 and digits >= len(t) * 0.34


def _metric_raw_ok(raw: str) -> bool:
    if "%" in raw:
        return False
    norm = cli.normalize_metric_text(raw)
    if not norm or not any(ch.isdigit() for ch in norm):
        return False
    if len(norm.replace("万", "").replace(".", "")) > 7:
        return False
    return cli.is_metric_value(raw)


def _infer_thumbnail_rect(title_item: dict, image_w: int, image_h: int) -> Dict[str, int]:
    """根据标题位置推断其上方封面（近似正方形）区域（双列信息流）。"""
    tr = title_item["rect"]
    col_w = max(image_w // 2, 1)
    margin_x = max(6, image_w // 90)
    inner_w = col_w - 2 * margin_x
    thumb_left = margin_x if int(tr["x"]) < col_w else col_w + margin_x
    title_top = int(tr["y"])
    gap = max(8, int(tr["height"]) // 3)
    thumb_bottom = min(image_h - 1, title_top - gap)
    side = max(40, min(inner_w, thumb_bottom))
    thumb_top = max(0, thumb_bottom - side)
    tw = min(inner_w, thumb_bottom - thumb_top)
    return {
        "x": thumb_left,
        "y": thumb_top,
        "width": max(10, tw),
        "height": max(10, thumb_bottom - thumb_top),
    }


def _overlay_strip_roi(thumb: Dict[str, int]) -> Dict[str, int]:
    """封面左下角小眼睛+浏览数字所在条带（相对封面区域）。"""
    w, h = int(thumb["width"]), int(thumb["height"])
    strip_h = max(22, min(44, int(h * 0.15)))
    strip_w = max(56, min(140, int(w * 0.52)))
    pad_x = max(3, int(w * 0.025))
    pad_y = max(3, int(h * 0.025))
    ty = int(thumb["y"]) + h - strip_h - pad_y
    return {
        "x": int(thumb["x"]) + pad_x,
        "y": max(0, ty),
        "width": strip_w,
        "height": strip_h,
    }


def _intersection_area(a: Dict[str, int], b: Dict[str, int]) -> int:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    return ix * iy


def _center_in_rect(rr: Dict[str, int], outer: Dict[str, int]) -> bool:
    cx = rr["x"] + rr["width"] / 2.0
    cy = rr["y"] + rr["height"] / 2.0
    return (
        outer["x"] <= cx <= outer["x"] + outer["width"]
        and outer["y"] <= cy <= outer["y"] + outer["height"]
    )


def _overlay_candidate_ok(raw: str, rr: Dict[str, int], roi: Dict[str, int]) -> bool:
    """条带内 OCR：纯数字优先；否则允许矮宽条内的合并杂框（封面底部叠字常见）。"""
    if not _metric_raw_ok(raw):
        return False
    if int(rr.get("height", 0)) < max(8, int(round(int(roi["height"]) * 0.22))):
        return False
    if _is_pure_views_numeric_ocr(raw):
        return True
    if _mixed_metric_fallback_ok(raw):
        return True
    inter = _intersection_area(rr, roi)
    if inter <= 0:
        return False
    item_area = max(1, rr["width"] * rr["height"])
    if inter < min(item_area * 0.12, 120):
        return False
    # 典型底部叠字：横向条，高度不大
    if rr["height"] <= 22 and rr["width"] <= 170:
        return True
    return False


def _pick_overlay_item(
    items: list,
    title_item: dict,
    thumb: Dict[str, int],
    roi: Dict[str, int],
    thumb_bottom: int,
) -> Optional[dict]:
    candidates: List[dict] = []
    for item in items:
        if item is title_item:
            continue
        raw = unicodedata.normalize("NFKC", str(item.get("text", ""))).strip()
        rr = item["rect"]
        if not _center_in_rect(rr, thumb):
            continue
        if _intersection_area(rr, roi) <= 0:
            continue
        if not _overlay_candidate_ok(raw, rr, roi):
            continue
        candidates.append(item)

    if not candidates:
        return None

    pure = [c for c in candidates if _is_pure_views_numeric_ocr(
        unicodedata.normalize("NFKC", str(c.get("text", ""))).strip()
    )]
    pool = pure if pure else candidates

    def bottom_snap_score(it: dict) -> Tuple[int, int]:
        r = it["rect"]
        bot = int(r["y"] + r["height"])
        return (abs(thumb_bottom - bot), int(r["width"]))

    pool.sort(key=bottom_snap_score)
    return pool[0]


def _roi_tuple(r: Dict[str, int]) -> Tuple[int, int, int, int]:
    return (r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])


def _bright_components(mask) -> list[Tuple[int, Tuple[int, int, int, int]]]:
    import numpy as np

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[Tuple[int, Tuple[int, int, int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            pts: list[Tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                pts.append((cx, cy))
                for ny in range(cy - 1, cy + 2):
                    for nx in range(cx - 1, cx + 2):
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                comps.append((len(pts), (min(xs), min(ys), max(xs) + 1, max(ys) + 1)))
    return comps


def _visual_overlay_item_from_roi(image: Image.Image, roi: Dict[str, int]) -> Optional[dict]:
    """Fallback when OCR misses a tiny view count: find bright digit strokes after the eye icon."""

    import numpy as np

    compact = dict(roi)
    compact["width"] = min(int(roi["width"]), 72)
    crop = image.crop(_roi_tuple(compact)).convert("RGB")
    arr = np.array(crop)
    if arr.size == 0:
        return None
    lum = arr.mean(axis=2)
    spread = arr.max(axis=2) - arr.min(axis=2)
    bg = float(np.median(lum))
    mask = (lum > max(185.0, bg + 28.0)) & (spread < 80)

    comps = _bright_components(mask)
    if not comps:
        return None

    h, w = mask.shape
    left_limit = max(18, int(w * 0.45))
    eye_like = []
    for area, box in comps:
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if x0 >= left_limit or area < 24 or bw < 6 or bh < 6:
            continue
        # Bright floor/background details can sit at the bottom of the strip and
        # are much wider than the eye mark. Treat only compact upper components
        # as the icon anchor; otherwise digit_min_x jumps past the real digits.
        if y1 > int(h * 0.74) or bw > 30 or bh > 24:
            continue
        eye_like.append((area, box))
    eye_right = max((box[2] for _area, box in eye_like), default=max(10, int(w * 0.26)))
    anchor_center_y = None
    if eye_like:
        _area, eye_box = max(eye_like, key=lambda item: item[0])
        anchor_center_y = compact["y"] + (eye_box[1] + eye_box[3]) / 2.0
    digit_min_x = eye_right + 3

    digit_comps = []
    for area, box in comps:
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if x0 < digit_min_x:
            continue
        if x0 > digit_min_x + 34:
            continue
        if y0 > int(h * 0.78):
            continue
        if bh < 6 or bw > 18 or area > 90:
            continue
        digit_comps.append((area, box))

    if not digit_comps:
        return None
    digit_comps.sort(key=lambda item: item[1][0])

    keep = [digit_comps[0][1]]
    last = keep[-1]
    for _area, box in digit_comps[1:]:
        gap = box[0] - last[2]
        if gap > 4:
            break
        if box[2] - keep[0][0] > 42:
            break
        keep.append(box)
        last = box

    x0 = max(0, min(box[0] for box in keep) - 1)
    y0 = max(0, min(box[1] for box in keep) - 1)
    x1 = min(w, max(box[2] for box in keep) + 1)
    y1 = min(h, max(box[3] for box in keep) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    width = max(8, x1 - x0)
    item = {
        "text": "1",
        "rect": {
            "x": compact["x"] + x0,
            "y": compact["y"] + y0,
            "width": width,
            "height": max(8, y1 - y0),
        },
    }
    if anchor_center_y is not None:
        item["overlay_anchor_center_y"] = anchor_center_y
    return item


def _refine_overlay_item_with_visual_digit(image: Image.Image, roi: Dict[str, int], item: dict) -> dict:
    """If OCR merged the eye icon into a one-digit box, keep text but use visual digit bounds."""

    raw = unicodedata.normalize("NFKC", str(item.get("text", ""))).strip()
    rr = item.get("rect", {})
    if not _is_pure_views_numeric_ocr(raw):
        return item
    digit_count = sum(1 for ch in raw if ch.isdigit())
    if digit_count != 1:
        return item
    width = int(rr.get("width", 0))
    height = max(1, int(rr.get("height", 1)))
    # Single overlay digits are narrow. A wide box generally means OCR included
    # the eye icon; rendering at that x puts the replacement on top of the icon.
    visual = _visual_overlay_item_from_roi(image, roi)
    if visual is None:
        return item

    vr = visual["rect"]
    # Single overlay digits are narrow. A wide OCR box generally means OCR
    # included the eye icon; conversely, a too-narrow OCR box may have captured
    # only the rightmost glyph of a two-digit number (e.g. "18" -> "8").
    wide_ocr_includes_icon = width > max(14, int(round(height * 1.05)))
    visual_extends_left = int(vr["x"]) < int(rr.get("x", 0)) - max(3, height // 4)
    visual_covers_ocr = int(vr["x"] + vr["width"]) >= int(rr.get("x", 0)) + max(2, width // 2)
    visual_is_wider = int(vr["width"]) > width + max(4, height // 3)
    if not (wide_ocr_includes_icon or (visual_extends_left and visual_covers_ocr and visual_is_wider)):
        return item
    refined = dict(item)
    refined["rect"] = dict(vr)
    refined["text"] = item.get("text", raw)
    if "overlay_anchor_center_y" in visual:
        refined["overlay_anchor_center_y"] = visual["overlay_anchor_center_y"]
    return refined


def find_card_eye_number_item(image: Image.Image, items: list, title_item: dict) -> dict:
    """定位封面左下角小眼睛旁浏览数字对应的 OCR 框（必要时退回整条几何 ROI）。"""
    w, h = image.size
    tr = title_item["rect"]
    thumb = _infer_thumbnail_rect(title_item, w, h)
    roi = _overlay_strip_roi(thumb)
    thumb_bottom = int(thumb["y"] + thumb["height"])

    picked = _pick_overlay_item(items, title_item, thumb, roi, thumb_bottom)
    if picked is not None:
        return _refine_overlay_item_with_visual_digit(image, roi, picked)

    # 条带内二次 OCR（叠字对比度差时全图 OCR 可能漏框）
    import numpy as np

    crop = image.crop(_roi_tuple(roi)).convert("RGB")
    local = cli.detect_items_with_paddle(np.array(crop))
    best: Optional[dict] = None
    best_w = 10**9
    for it in local:
        raw = unicodedata.normalize("NFKC", str(it.get("text", ""))).strip()
        if not _metric_raw_ok(raw):
            continue
        rr = dict(it["rect"])
        rr["x"] += roi["x"]
        rr["y"] += roi["y"]
        if not _overlay_candidate_ok(raw, rr, roi):
            continue
        if _is_pure_views_numeric_ocr(raw):
            return _refine_overlay_item_with_visual_digit(image, roi, {"text": it["text"], "rect": rr})
        if rr["width"] < best_w:
            best_w = rr["width"]
            best = {"text": it["text"], "rect": rr}
    if best is not None:
        return best

    visual = _visual_overlay_item_from_roi(image, roi)
    if visual is not None:
        return visual

    raise RuntimeError(
        "未在封面左下角条带内识别到浏览数字。"
        "请确认截图含完整封面与标题，或换更清晰图片。"
    )


def _longest_digit_run_length(raw: str) -> int:
    """原始 OCR 里最长连续 ASCII 数字长度（应对规范化只抽到其中一位的情况）。"""
    t = unicodedata.normalize("NFKC", str(raw))
    best = cur = 0
    for ch in t:
        if ch.isdigit():
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _infer_digit_slots_from_views_ocr_rect(rect: Dict[str, int]) -> int:
    """由封面浏览数字 OCR 框宽高推断位数（漏识别 ``40→4`` 时框往往仍覆盖两位数宽度）。"""
    w = max(1.0, float(rect["width"]))
    h = max(1.0, float(rect["height"]))
    # 胶囊内数字水平占位约为字高的 ~0.65–0.75；用 0.72。
    est = int(round(w / max(h * 0.72, 1e-6)))
    return max(1, min(7, est))


def _overlay_digit_slot_count(
    original_norm: str,
    raw_ocr_hint: str = "",
    *,
    ocr_rect: Optional[Dict[str, int]] = None,
) -> int:
    """位数：规范化数字个数与 OCR 原文最长连续数字段取较大者。

    若二者都只有 1 位，再用 OCR 框宽高推断（应对 ``40`` 被识别成单个 ``4`` 且原文也无 ``0``）。
    """

    n_norm = sum(1 for c in original_norm if c.isdigit())
    n_raw = _longest_digit_run_length(raw_ocr_hint) if raw_ocr_hint else 0
    n_text = max(n_norm, n_raw)
    if n_text >= 2:
        return max(1, min(7, n_text))
    if ocr_rect is None:
        return 1
    geo = _infer_digit_slots_from_views_ocr_rect(ocr_rect)
    return max(1, min(7, geo))


def clamp_feed_overlay_views_to_digit_slots(
    original_norm: str,
    requested: str,
    *,
    raw_ocr_hint: str = "",
    ocr_rect: Optional[Dict[str, int]] = None,
) -> Tuple[str, Optional[str]]:
    """按原浏览数字位数限制新值：*n* 位 → 最大值 ``10**n - 1``（如一位 ``9``、两位 ``99``）。

    位数参考规范化数字、OCR 原文最长连续数字段；若仍只有一位则参考 OCR 框宽高（漏识别两位数时常框仍较宽）。
    """

    slots = _overlay_digit_slot_count(original_norm, raw_ocr_hint, ocr_rect=ocr_rect)
    max_val = 10**slots - 1

    req_norm = cli.normalize_metric_text(requested)
    digits = "".join(c for c in req_norm if c.isdigit())
    if not digits:
        raise ValueError("新浏览量需包含至少一位数字")

    val = int(digits)
    val = max(0, min(val, max_val))
    result = str(val)
    if int(digits) > max_val:
        hint = (
            f"原浏览量为 {slots} 位数字（最大 {max_val}），输入已超过上限，已改为 {result}"
        )
        return result, hint
    return result, None


def _feed_overlay_slot_cap(
    original_norm: str,
    *,
    raw_ocr_hint: str = "",
    ocr_rect: Optional[Dict[str, int]] = None,
) -> Tuple[int, int]:
    slots = _overlay_digit_slot_count(original_norm, raw_ocr_hint, ocr_rect=ocr_rect)
    return slots, 10**slots - 1


def choose_feed_overlay_views_for_slots(
    original_norm: str,
    requested_range: Tuple[int, int],
    *,
    raw_ocr_hint: str = "",
    ocr_rect: Optional[Dict[str, int]] = None,
    rng=random,
) -> Tuple[str, Optional[str]]:
    """Pick a random view count that still fits the original overlay digit slots."""

    lo, hi = requested_range
    if lo > hi:
        lo, hi = hi, lo
    slots, max_val = _feed_overlay_slot_cap(
        original_norm,
        raw_ocr_hint=raw_ocr_hint,
        ocr_rect=ocr_rect,
    )
    bounded_hi = min(hi, max_val)
    bounded_lo = min(lo, bounded_hi)
    result = str(rng.randint(bounded_lo, bounded_hi))
    if hi > max_val:
        return (
            result,
            f"原浏览量为 {slots} 位数字（最大 {max_val}），随机范围已收窄为 {bounded_lo}-{bounded_hi}",
        )
    return result, None


def beautify_feed_card_eye(
    args: SimpleNamespace,
    *,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Image.Image:
    """ args 需含: normal, eye_title (str), eye_views (str), 以及 ocr/glyph_atlas 等。 """
    title_query = str(getattr(args, "eye_title", "") or "").strip()
    new_views = str(getattr(args, "eye_views", "") or "").strip()
    has_views_range = hasattr(args, "eye_views_range")
    if not new_views and not has_views_range:
        raise ValueError("新浏览量不能为空")

    def _prog(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    _prog("加载图片…")
    image = Image.open(args.normal).convert("RGB")
    source_image = image.copy()

    import numpy as np

    _prog("OCR 识别文字…")
    items = cli.detect_items_with_paddle(np.array(image))
    fallback_atlas = cli.build_glyph_atlas(source_image, items) if getattr(args, "glyph_atlas", False) else {}

    _prog("匹配标题…")
    title_item, score = pick_best_title_item(items, title_query)
    _prog(f"标题匹配 score={score:.2f} → {title_item['text']!r}")

    _prog("定位小眼睛数字…")
    view_item = find_card_eye_number_item(image, items, title_item)
    rect = dict(view_item["rect"])
    original_text = cli.normalize_metric_text(view_item["text"])

    thumb = _infer_thumbnail_rect(title_item, image.width, image.height)
    strip_roi = _overlay_strip_roi(thumb)
    slot_rect = cli.localize_feed_overlay_views_ink(
        source_image,
        rect,
        raw_text=str(view_item["text"]),
    ) or rect

    if has_views_range:
        clamped_views, clamp_hint = choose_feed_overlay_views_for_slots(
            original_text,
            tuple(getattr(args, "eye_views_range")),
            raw_ocr_hint=str(view_item["text"]),
            ocr_rect=slot_rect,
        )
    else:
        clamped_views, clamp_hint = clamp_feed_overlay_views_to_digit_slots(
            original_text,
            new_views,
            raw_ocr_hint=str(view_item["text"]),
            ocr_rect=slot_rect,
        )
    if clamp_hint:
        _prog(clamp_hint)

    if not cli.metric_text_changed(original_text, clamped_views):
        _prog("数字相同，跳过")
        return image

    _prog("渲染新数字…")
    image, _report = cli.patch_ocr_rect_with_glyphs(
        image,
        source_image,
        rect,
        clamped_views,
        fallback_atlas,
        original_text,
        row_atlas=None,
        raw_text=view_item["text"],
        forced_font=None,
        overlay_anchor_center_y=view_item.get("overlay_anchor_center_y"),
        overlay_views_ink=True,
        overlay_thumb=thumb,
        overlay_strip=strip_roi,
    )
    _prog("完成")
    return image
