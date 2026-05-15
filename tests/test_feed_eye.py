import unittest
from PIL import Image

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

    def test_raises_when_no_overlay_candidate(self):
        title = {"text": "只有标题足够长度中文", "rect": {"x": 20, "y": 400, "width": 180, "height": 22}}
        img = Image.new("RGB", (540, 700), color=(255, 255, 255))
        items = [title]
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


if __name__ == "__main__":
    unittest.main()
