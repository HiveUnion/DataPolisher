import unittest

from PIL import Image, ImageDraw

from data_polisher.core import calculate_metrics, detect_dark_text_bounds
from data_polisher.cli import (
    inpaint_overlay_views_stroke_fill,
    normalize_metric_text,
    segment_glyph_boxes,
    style_distance,
)
from data_polisher.ocr import extract_ocr_text_bounds


class MetricsTest(unittest.TestCase):
    def test_calculates_click_and_interaction_rates(self):
        metrics = calculate_metrics(
            exposure=200,
            views=100,
            likes=10,
            comments=5,
            collects=3,
            shares=2,
        )

        self.assertEqual(metrics["click_rate_text"], "50%")
        self.assertEqual(metrics["interaction_rate_text"], "20%")

    def test_rejects_non_positive_exposure(self):
        with self.assertRaises(ValueError):
            calculate_metrics(exposure=0, views=100, likes=1, comments=1, collects=1, shares=1)


class TextBoundsTest(unittest.TestCase):
    def test_detects_dark_text_bounds_in_candidate_region(self):
        pixels = [[(255, 255, 255) for _ in range(10)] for _ in range(8)]
        for y in range(2, 6):
            for x in range(3, 7):
                pixels[y][x] = (35, 35, 35)

        bounds = detect_dark_text_bounds(pixels)

        self.assertEqual(bounds, {"x": 3, "y": 2, "width": 4, "height": 4})

    def test_returns_none_when_no_text_pixels_exist(self):
        pixels = [[(255, 255, 255) for _ in range(5)] for _ in range(5)]

        self.assertIsNone(detect_dark_text_bounds(pixels))


class OcrBoundsTest(unittest.TestCase):
    def test_extracts_bounds_from_paddleocr_result(self):
        result = [
            [
                [
                    [[3, 2], [8, 2], [8, 10], [3, 10]],
                    ("122", 0.99),
                ],
                [
                    [[20, 2], [35, 2], [35, 9], [20, 9]],
                    ("粉丝占0%", 0.97),
                ],
            ]
        ]

        self.assertEqual(extract_ocr_text_bounds(result), {"x": 3, "y": 2, "width": 32, "height": 8})

    def test_ignores_non_metric_text(self):
        result = [[[[[1, 1], [20, 1], [20, 8], [1, 8]], ("曝光数", 0.98)]]]

        self.assertIsNone(extract_ocr_text_bounds(result))


class MetricTextTest(unittest.TestCase):
    def test_normalizes_metric_text_to_supported_glyphs(self):
        self.assertEqual(normalize_metric_text("18.2%"), "18.2%")
        self.assertEqual(normalize_metric_text("粉丝占0%"), "0%")
        self.assertEqual(normalize_metric_text(" 36 "), "36")

    def test_style_distance_scores_identical_style_as_zero(self):
        style = {
            "density": 0.5,
            "edge_ratio": 0.2,
            "alpha_summary": {"p10": 20, "p50": 220, "p90": 255},
        }

        self.assertEqual(style_distance(style, style), 0)


class GlyphSegmentationTest(unittest.TestCase):
    def test_segments_three_digits_separated_by_clear_gaps(self):
        pixels = [[(255, 255, 255) for _ in range(9)] for _ in range(5)]
        for row in range(5):
            for x in (0, 1, 4, 5, 7, 8):
                pixels[row][x] = (40, 40, 40)

        boxes = segment_glyph_boxes(pixels)

        self.assertEqual(
            boxes,
            [
                {"x": 0, "y": 0, "width": 2, "height": 5},
                {"x": 4, "y": 0, "width": 2, "height": 5},
                {"x": 7, "y": 0, "width": 2, "height": 5},
            ],
        )


class OverlayStrokeInpaintTests(unittest.TestCase):
    def test_erases_bright_core_and_antialias_halo(self):
        image = Image.new("RGB", (64, 36), (120, 72, 42))
        draw = ImageDraw.Draw(image)
        for x in range(64):
            color = (110 + x // 3, 62 + x // 5, 38 + x // 7)
            draw.line((x, 0, x, 35), fill=color)
        draw.rounded_rectangle((6, 8, 48, 28), radius=10, fill=(88, 62, 48))
        before_bg = image.getpixel((26, 18))
        draw.line((25, 10, 25, 25), fill=(190, 190, 184), width=5)
        draw.line((25, 10, 25, 25), fill=(255, 255, 255), width=3)

        patched = inpaint_overlay_views_stroke_fill(
            image,
            image,
            {"x": 20, "y": 8, "width": 12, "height": 22},
        )

        center = patched.getpixel((25, 18))
        halo = patched.getpixel((23, 18))
        self.assertLess(sum(center) / 3, 170)
        self.assertLess(sum(halo) / 3, 170)
        self.assertLess(abs(center[0] - before_bg[0]), 55)
        self.assertEqual(patched.getpixel((4, 4)), image.getpixel((4, 4)))

    def test_erases_tiny_antialias_fragments_in_digit_lane(self):
        image = Image.new("RGB", (64, 36), (128, 140, 148))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((6, 8, 50, 29), radius=10, fill=(118, 126, 132))
        before_bg = image.getpixel((34, 24))
        draw.line((30, 12, 30, 23), fill=(255, 255, 255), width=2)
        # Low, disconnected JPEG-like edge fragments used to be filtered out by
        # component height and survived below the replacement digits.
        for pt in ((29, 25), (30, 26), (36, 25)):
            draw.point(pt, fill=(176, 178, 178))

        patched = inpaint_overlay_views_stroke_fill(
            image,
            image,
            {"x": 26, "y": 10, "width": 14, "height": 19},
        )

        for pt in ((29, 25), (30, 26), (36, 25)):
            px = patched.getpixel(pt)
            self.assertLess(abs(px[0] - before_bg[0]), 45)
            self.assertLess(abs(px[1] - before_bg[1]), 45)
            self.assertLess(abs(px[2] - before_bg[2]), 45)

    def test_erases_dim_compression_residue_between_digit_strokes(self):
        image = Image.new("RGB", (64, 36), (128, 140, 148))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((6, 8, 52, 29), radius=10, fill=(118, 126, 132))
        before_bg = image.getpixel((33, 18))
        draw.line((29, 12, 29, 23), fill=(255, 255, 255), width=2)
        draw.line((38, 12, 38, 23), fill=(255, 255, 255), width=2)
        draw.point((33, 18), fill=(150, 152, 152))

        patched = inpaint_overlay_views_stroke_fill(
            image,
            image,
            {"x": 26, "y": 10, "width": 16, "height": 18},
        )

        px = patched.getpixel((33, 18))
        self.assertLess(abs(px[0] - before_bg[0]), 45)
        self.assertLess(abs(px[1] - before_bg[1]), 45)
        self.assertLess(abs(px[2] - before_bg[2]), 45)


class HeaderViewPickTests(unittest.TestCase):
    """顶部统计行：从左第一个数字为小眼睛（排除左侧日期）。"""

    def test_find_header_view_value_first_metric_left_of_row(self):
        from data_polisher.cli import find_header_view_value, normalize_metric_text

        items = [
            {"text": "05-07", "rect": {"x": 37, "y": 232, "width": 55, "height": 14}},
            {"text": "◎6", "rect": {"x": 137, "y": 234, "width": 37, "height": 15}},
            {"text": "0", "rect": {"x": 283, "y": 233, "width": 11, "height": 13}},
        ]
        picked = find_header_view_value(items, image_size=(540, 1200))
        self.assertEqual(normalize_metric_text(picked["text"]), "6")

    def test_find_header_skips_date_before_eye_metric(self):
        from data_polisher.cli import find_header_view_value, normalize_metric_text

        items = [
            {"text": "04-27", "rect": {"x": 40, "y": 233, "width": 50, "height": 14}},
            {"text": "◎ 36", "rect": {"x": 138, "y": 233, "width": 50, "height": 17}},
            {"text": "0", "rect": {"x": 293, "y": 234, "width": 12, "height": 15}},
        ]
        picked = find_header_view_value(items, image_size=(540, 1200))
        self.assertEqual(normalize_metric_text(picked["text"]), "36")


if __name__ == "__main__":
    unittest.main()
