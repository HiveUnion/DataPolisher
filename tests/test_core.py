import unittest

from data_polisher.core import calculate_metrics, detect_dark_text_bounds
from data_polisher.cli import normalize_metric_text, segment_glyph_boxes, style_distance
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


if __name__ == "__main__":
    unittest.main()
