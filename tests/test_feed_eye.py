import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from data_polisher import cli
from data_polisher import feed_eye


class FeedEyeTitlePickTests(unittest.TestCase):
    def test_substring_prefers_contained_query(self):
        items = [
            {"text": "神明也爱住城中村？", "rect": {"x": 0, "y": 0, "width": 10, "height": 10}},
            {"text": "其他标题", "rect": {"x": 0, "y": 100, "width": 10, "height": 10}},
        ]
        best, score = feed_eye.pick_best_title_item(items, "神明")
        self.assertIn("神明", best["text"])
        self.assertGreater(score, 0.9)

    def test_query_can_include_date_hint(self):
        items = [
            {"text": "谁懂啊！有线耳机用一个", "rect": {"x": 286, "y": 941, "width": 220, "height": 24}},
            {"text": "昨天18:59", "rect": {"x": 319, "y": 1019, "width": 61, "height": 17}},
        ]
        best, score = feed_eye.pick_best_title_item(items, "耳机 12-23")
        self.assertIn("耳机", best["text"])
        self.assertGreater(score, 0.8)

    def test_raises_when_no_title_candidate(self):
        items = [{"text": "123", "rect": {"x": 0, "y": 0, "width": 1, "height": 1}}]
        with self.assertRaises(RuntimeError):
            feed_eye.pick_best_title_item(items, "任意")


class FeedEyeThumbnailOverlayTests(unittest.TestCase):
    """封面左下角小眼睛条带（根据标题推断栅格列与封面区域）。"""

    def _img(self):
        return Image.new("RGB", (540, 1200), color=(240, 240, 240))

    def test_left_column_pure_views_in_overlay_roi(self):
        title = {"text": "神明也爱住城中村？", "rect": {"x": 17, "y": 974, "width": 180, "height": 24}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        inner = {
            "text": "88",
            "rect": {
                "x": roi["x"] + 30,
                "y": roi["y"] + 4,
                "width": 22,
                "height": 16,
            },
        }
        img = self._img()
        items = [title, inner]
        it = feed_eye.find_card_eye_number_item(img, items, title)
        self.assertEqual(it["text"], "88")

    def test_merged_low_strip_accepted_when_short_height(self):
        """封面底部叠字：横向宽框但高度很矮时可作为候选。"""
        title = {"text": "神明也爱住城中村？", "rect": {"x": 17, "y": 974, "width": 180, "height": 24}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        merged = {
            "text": "足犹40尽这么多",
            "rect": {
                "x": roi["x"] + 4,
                "y": roi["y"] + 2,
                "width": 130,
                "height": 13,
            },
        }
        img = self._img()
        items = [title, merged]
        it = feed_eye.find_card_eye_number_item(img, items, title)
        self.assertEqual(cli.normalize_metric_text(it["text"]), "40")

    def test_ignores_metrics_outside_thumbnail(self):
        title = {"text": "神明也爱住城中村？", "rect": {"x": 17, "y": 974, "width": 180, "height": 24}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        good = {
            "text": "36",
            "rect": {"x": roi["x"] + 40, "y": roi["y"] + 4, "width": 20, "height": 14},
        }
        bad_below = {
            "text": "999",
            "rect": {"x": 200, "y": title["rect"]["y"] + 40, "width": 28, "height": 14},
        }
        img = self._img()
        items = [title, good, bad_below]
        it = feed_eye.find_card_eye_number_item(img, items, title)
        self.assertEqual(it["text"], "36")

    def test_right_column_title(self):
        title = {"text": "减脂期笔记标题示例文字", "rect": {"x": 286, "y": 974, "width": 220, "height": 24}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        views = {
            "text": "17",
            "rect": {"x": roi["x"] + 10, "y": roi["y"] + 5, "width": 18, "height": 14},
        }
        img = self._img()
        items = [title, views]
        it = feed_eye.find_card_eye_number_item(img, items, title)
        self.assertEqual(it["text"], "17")

    def test_prefers_pure_numeric_in_strip_over_wide_merged(self):
        title = {"text": "标题文案足够长度中文", "rect": {"x": 20, "y": 500, "width": 200, "height": 22}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 900)
        roi = feed_eye._overlay_strip_roi(thumb)
        merged = {
            "text": "混40乱字",
            "rect": {"x": roi["x"] + 2, "y": roi["y"] + 1, "width": 100, "height": 14},
        }
        pure = {
            "text": "99",
            "rect": {"x": roi["x"] + 85, "y": roi["y"] + 2, "width": 22, "height": 14},
        }
        img = Image.new("RGB", (540, 900), color=(200, 200, 200))
        items = [title, merged, pure]
        it = feed_eye.find_card_eye_number_item(img, items, title)
        self.assertEqual(it["text"], "99")

    def test_visual_fallback_finds_tiny_one_when_ocr_misses_overlay(self):
        title = {"text": "谁懂啊！有线耳机用一个", "rect": {"x": 285, "y": 941, "width": 228, "height": 24}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(170, 145, 110),
        )
        # Semi-transparent-looking pill, eye icon, and a very small one-digit count.
        draw.rounded_rectangle(
            (roi["x"] + 6, roi["y"] + 7, roi["x"] + 54, roi["y"] + 30),
            radius=12,
            fill=(120, 112, 98),
        )
        draw.ellipse((roi["x"] + 14, roi["y"] + 14, roi["x"] + 30, roi["y"] + 23), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 20, roi["y"] + 16, roi["x"] + 24, roi["y"] + 20), fill=(120, 112, 98))
        draw.line((roi["x"] + 39, roi["y"] + 10, roi["x"] + 39, roi["y"] + 24), fill=(255, 255, 255), width=2)

        with patch("data_polisher.feed_eye.cli.detect_items_with_paddle", return_value=[]):
            it = feed_eye.find_card_eye_number_item(img, [title], title)

        self.assertEqual(it["text"], "1")
        self.assertGreaterEqual(it["rect"]["x"], roi["x"] + 35)
        self.assertLess(it["rect"]["x"], roi["x"] + 48)

    def test_refines_single_digit_ocr_box_that_includes_eye_icon(self):
        title = {"text": "实测音质不输大牌", "rect": {"x": 286, "y": 836, "width": 220, "height": 24}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(165, 125, 90),
        )
        draw.rounded_rectangle(
            (roi["x"] + 8, roi["y"] + 6, roi["x"] + 58, roi["y"] + 29),
            radius=12,
            fill=(112, 104, 92),
        )
        draw.ellipse((roi["x"] + 14, roi["y"] + 13, roi["x"] + 30, roi["y"] + 22), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 20, roi["y"] + 15, roi["x"] + 24, roi["y"] + 20), fill=(112, 104, 92))
        draw.text((roi["x"] + 38, roi["y"] + 7), "2", fill=(255, 255, 255))
        merged = {
            "text": "2",
            "rect": {"x": roi["x"] + 12, "y": roi["y"] + 5, "width": 30, "height": 18},
        }

        with patch("data_polisher.feed_eye.cli.detect_items_with_paddle", return_value=[]):
            it = feed_eye.find_card_eye_number_item(img, [title, merged], title)

        self.assertEqual(it["text"], "2")
        self.assertGreaterEqual(it["rect"]["x"], roi["x"] + 34)
        self.assertLess(it["rect"]["x"], roi["x"] + 48)

    def test_raises_when_no_overlay_candidate(self):
        title = {"text": "只有标题足够长度中文", "rect": {"x": 20, "y": 400, "width": 180, "height": 22}}
        img = Image.new("RGB", (540, 700), color=(255, 255, 255))
        items = [title]
        with patch("data_polisher.feed_eye.cli.detect_items_with_paddle", return_value=[]):
            with self.assertRaises(RuntimeError):
                feed_eye.find_card_eye_number_item(img, items, title)


class FeedEyeClampViewsTests(unittest.TestCase):
    def test_one_digit_max_9(self):
        s, hint = feed_eye.clamp_feed_overlay_views_to_digit_slots("5", "42")
        self.assertEqual(s, "9")
        self.assertIsNotNone(hint)

    def test_raw_ocr_longest_run_used_when_norm_single_digit(self):
        """原文「…40…」时不能只凭规范化成的一位限制为 9。"""
        s, hint = feed_eye.clamp_feed_overlay_views_to_digit_slots(
            "4",
            "100",
            raw_ocr_hint="足。犹40尽这么多",
        )
        self.assertEqual(s, "99")
        self.assertIsNotNone(hint)

    def test_two_digit_max_99(self):
        s, hint = feed_eye.clamp_feed_overlay_views_to_digit_slots("40", "88888")
        self.assertEqual(s, "99")
        self.assertIsNotNone(hint)

    def test_three_digit_under_cap_no_hint(self):
        s, hint = feed_eye.clamp_feed_overlay_views_to_digit_slots("147", "500")
        self.assertEqual(s, "500")
        self.assertIsNone(hint)

    def test_bbox_infer_two_slots_when_ocr_single_digit_wide_box(self):
        """OCR 文本仅有「4」但框宽度约为两位数时，应按宽高推断为两位上限。"""
        s, hint = feed_eye.clamp_feed_overlay_views_to_digit_slots(
            "4",
            "100",
            raw_ocr_hint="4",
            ocr_rect={"x": 0, "y": 0, "width": 22, "height": 14},
        )
        self.assertEqual(s, "99")
        self.assertIsNotNone(hint)

    def test_bbox_keeps_one_slot_when_narrow_box_and_ocr_single_digit(self):
        s, hint = feed_eye.clamp_feed_overlay_views_to_digit_slots(
            "4",
            "100",
            raw_ocr_hint="4",
            ocr_rect={"x": 0, "y": 0, "width": 10, "height": 14},
        )
        self.assertEqual(s, "9")
        self.assertIsNotNone(hint)

    def test_range_is_bounded_to_original_digit_slots_before_random(self):
        class MaxRng:
            def randint(self, lo, hi):
                return hi

        s, hint = feed_eye.choose_feed_overlay_views_for_slots(
            "40",
            (80, 120),
            raw_ocr_hint="40",
            ocr_rect={"x": 0, "y": 0, "width": 22, "height": 14},
            rng=MaxRng(),
        )
        self.assertEqual(s, "99")
        self.assertIn("80-99", hint or "")

    def test_range_uses_geometric_slots_when_ocr_drops_a_digit(self):
        class MaxRng:
            def randint(self, lo, hi):
                return hi

        s, hint = feed_eye.choose_feed_overlay_views_for_slots(
            "4",
            (80, 120),
            raw_ocr_hint="4",
            ocr_rect={"x": 0, "y": 0, "width": 22, "height": 14},
            rng=MaxRng(),
        )
        self.assertEqual(s, "99")
        self.assertIsNotNone(hint)

    def test_range_respects_requested_upper_bound_when_it_fits_slots(self):
        class MaxRng:
            def randint(self, lo, hi):
                return hi

        s, hint = feed_eye.choose_feed_overlay_views_for_slots(
            "40",
            (40, 69),
            raw_ocr_hint="40",
            ocr_rect={"x": 0, "y": 0, "width": 22, "height": 14},
            rng=MaxRng(),
        )
        self.assertEqual(s, "69")
        self.assertIsNone(hint)


if __name__ == "__main__":
    unittest.main()
