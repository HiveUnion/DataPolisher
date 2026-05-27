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

    def test_prefers_feed_title_over_cover_text_for_same_keyword(self):
        items = [
            {"text": "护耳式", "rect": {"x": 35, "y": 331, "width": 69, "height": 25}},
            {"text": "儿童耳机", "rect": {"x": 54, "y": 352, "width": 95, "height": 31}},
            {"text": "送侄子的儿童耳机到了，", "rect": {"x": 20, "y": 584, "width": 211, "height": 21}},
            {"text": "小家伙爱不释手！", "rect": {"x": 19, "y": 612, "width": 153, "height": 21}},
            {"text": "昨天17:40", "rect": {"x": 53, "y": 664, "width": 59, "height": 14}},
        ]

        best, score = feed_eye.pick_best_title_item(items, "耳机")

        self.assertEqual(best["text"], "送侄子的儿童耳机到了，")
        self.assertGreater(score, 0.9)

    def test_keyword_on_second_title_line_uses_block_top(self):
        items = [
            {"text": "我没办法了。。失眠。失", "rect": {"x": 287, "y": 660, "width": 224, "height": 21}},
            {"text": "眠。还是失眠。", "rect": {"x": 284, "y": 684, "width": 136, "height": 27}},
            {"text": "我头", "rect": {"x": 315, "y": 714, "width": 37, "height": 23}},
            {"text": "昨天18:54", "rect": {"x": 316, "y": 735, "width": 65, "height": 18}},
        ]

        best, score = feed_eye.pick_best_title_item(items, "失眠")

        self.assertIn("我没办法", best["text"])
        self.assertIn("还是失眠", best["text"])
        self.assertEqual(best["rect"]["y"], 660)
        self.assertGreater(score, 0.9)

    def test_short_cjk_query_tolerates_one_ocr_missed_character(self):
        items = [
            {"text": "成都", "rect": {"x": 344, "y": 954, "width": 101, "height": 56}},
            {"text": "三城漫游|赴河，烟", "rect": {"x": 287, "y": 1018, "width": 224, "height": 21}},
        ]

        best, score = feed_eye.pick_best_title_item(items, "山河")

        self.assertEqual(best["text"], "三城漫游|赴河，烟")
        self.assertGreater(score, 0.35)

    def test_duplicate_cover_and_feed_title_candidates_try_lower_first(self):
        items = [
            {"text": "AI客服体验", "rect": {"x": 302, "y": 717, "width": 172, "height": 34}},
            {"text": "升级，声网", "rect": {"x": 301, "y": 760, "width": 163, "height": 37}},
            {"text": "助力服务提效", "rect": {"x": 304, "y": 805, "width": 190, "height": 31}},
            {"text": "AI客服体验升级，声网助", "rect": {"x": 286, "y": 967, "width": 226, "height": 22}},
            {"text": "力服务提效", "rect": {"x": 287, "y": 994, "width": 103, "height": 23}},
        ]

        candidates = feed_eye.pick_title_candidates_for_eye(items, "声网", (540, 1200))

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0][0]["rect"]["y"], 967)
        self.assertIn("声网", candidates[0][0]["text"])
        self.assertEqual(candidates[1][0]["rect"]["y"], 717)

    def test_title_candidate_lookup_falls_back_to_upper_when_lower_has_no_eye(self):
        img = Image.new("RGB", (540, 1200), color=(255, 255, 255))
        lower = {"text": "下方声网标题", "rect": {"x": 286, "y": 967, "width": 220, "height": 24}}
        upper = {"text": "封面声网标题", "rect": {"x": 302, "y": 717, "width": 170, "height": 80}}
        view_item = {"text": "23", "rect": {"x": 320, "y": 916, "width": 20, "height": 14}}

        with patch(
            "data_polisher.feed_eye.find_card_eye_number_item",
            side_effect=[RuntimeError("miss"), view_item],
        ) as mocked:
            title, found = feed_eye.find_eye_number_with_title_candidates(
                img,
                [],
                [(lower, 1.0), (upper, 0.98)],
            )

        self.assertIs(title, upper)
        self.assertEqual(found, view_item)
        self.assertIs(mocked.call_args_list[0].args[2], lower)
        self.assertIs(mocked.call_args_list[1].args[2], upper)

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

    def test_keyword_matches_bluetooth_title_and_picks_two_digit_overlay(self):
        items = [
            {
                "text": "韶雅S1睡眠蓝牙音响：黑",
                "rect": {"x": 286, "y": 972, "width": 220, "height": 24},
            },
        ]
        title, score = feed_eye.pick_best_title_item(items, "蓝牙")
        self.assertGreater(score, 0.9)

        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        views = {
            "text": "10",
            "rect": {"x": roi["x"] + 36, "y": roi["y"] + 6, "width": 22, "height": 15},
        }
        img = self._img()
        it = feed_eye.find_card_eye_number_item(img, [title, views], title)

        self.assertEqual(it["text"], "10")

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

    def test_refines_single_digit_ocr_box_that_missed_leading_digit(self):
        title = {"text": "失眠救星韶雅骨传导睡", "rect": {"x": 286, "y": 987, "width": 220, "height": 20}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(164, 132, 112),
        )
        draw.rounded_rectangle(
            (roi["x"] + 5, roi["y"] + 2, roi["x"] + 68, roi["y"] + 25),
            radius=12,
            fill=(132, 118, 104),
        )
        # A bright texture at the bottom of the ROI should not be treated as
        # the eye icon anchor.
        draw.rectangle((roi["x"] + 24, roi["y"] + 28, roi["x"] + 72, roi["y"] + 37), fill=(230, 230, 226))
        draw.ellipse((roi["x"] + 13, roi["y"] + 6, roi["x"] + 31, roi["y"] + 19), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 20, roi["y"] + 10, roi["x"] + 25, roi["y"] + 15), fill=(132, 118, 104))
        draw.line((roi["x"] + 38, roi["y"] + 6, roi["x"] + 38, roi["y"] + 18), fill=(255, 255, 255), width=3)
        draw.ellipse((roi["x"] + 44, roi["y"] + 6, roi["x"] + 51, roi["y"] + 12), outline=(255, 255, 255), width=2)
        draw.ellipse((roi["x"] + 44, roi["y"] + 12, roi["x"] + 51, roi["y"] + 19), outline=(255, 255, 255), width=2)
        missed_leading = {
            "text": "8",
            "rect": {"x": roi["x"] + 44, "y": roi["y"] + 7, "width": 8, "height": 12},
        }

        it = feed_eye._refine_overlay_item_with_visual_digit(img, roi, missed_leading)

        self.assertEqual(it["text"], "18")
        self.assertLessEqual(it["rect"]["x"], roi["x"] + 38)
        self.assertGreaterEqual(it["rect"]["width"], 14)

    def test_visual_fallback_recognizes_zero_digit(self):
        title = {"text": "睡眠蓝牙音箱", "rect": {"x": 286, "y": 987, "width": 220, "height": 20}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(150, 128, 112),
        )
        draw.rounded_rectangle(
            (roi["x"] + 5, roi["y"] + 2, roi["x"] + 64, roi["y"] + 25),
            radius=12,
            fill=(128, 116, 104),
        )
        draw.ellipse((roi["x"] + 13, roi["y"] + 6, roi["x"] + 31, roi["y"] + 19), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 20, roi["y"] + 10, roi["x"] + 25, roi["y"] + 15), fill=(128, 116, 104))
        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 17)
        draw.text((roi["x"] + 37, roi["y"] + 1), "0", fill=(255, 255, 255), font=font)

        it = feed_eye._visual_overlay_item_from_roi(img, roi)

        self.assertIsNotNone(it)
        self.assertEqual(it["text"], "0")

    def test_visual_fallback_handles_pale_cover_around_capsule(self):
        title = {"text": "2026春招盯紧这波大厂！", "rect": {"x": 19, "y": 943, "width": 216, "height": 21}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(236, 229, 210))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(
            (roi["x"] + 8, roi["y"] + 1, roi["x"] + 57, roi["y"] + 26),
            radius=13,
            fill=(190, 190, 184),
        )
        draw.ellipse((roi["x"] + 17, roi["y"] + 6, roi["x"] + 35, roi["y"] + 18), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 24, roi["y"] + 10, roi["x"] + 29, roi["y"] + 15), fill=(190, 190, 184))
        draw.line((roi["x"] + 42, roi["y"] + 6, roi["x"] + 42, roi["y"] + 18), fill=(255, 255, 255), width=3)

        it = feed_eye._visual_overlay_item_from_roi(img, roi)

        self.assertIsNotNone(it)
        self.assertEqual(it["text"], "1")
        self.assertIn("overlay_anchor_center_y", it)

    def test_visual_fallback_uses_lower_threshold_for_faint_two_digits(self):
        title = {"text": "连续一周睡不下了", "rect": {"x": 18, "y": 628, "width": 165, "height": 23}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(231, 234, 226))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(231, 234, 226),
        )
        draw.rounded_rectangle(
            (roi["x"] + 6, roi["y"] + 2, roi["x"] + 66, roi["y"] + 26),
            radius=12,
            fill=(174, 184, 167),
        )
        draw.ellipse((roi["x"] + 17, roi["y"] + 8, roi["x"] + 35, roi["y"] + 20), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 23, roi["y"] + 11, roi["x"] + 28, roi["y"] + 16), fill=(174, 184, 167))
        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 17)
        draw.text((roi["x"] + 42, roi["y"] + 3), "20", fill=(235, 235, 235), font=font)

        it = feed_eye._visual_overlay_item_from_roi(img, roi, expected_digit_count=2)

        self.assertIsNotNone(it)
        self.assertEqual(it["text"], "20")

    def test_refines_two_digit_ocr_with_matching_visual_anchor(self):
        title = {"text": "骨传导睡眠音箱-让枕头", "rect": {"x": 20, "y": 942, "width": 213, "height": 21}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(112, 78, 38),
        )
        draw.rounded_rectangle(
            (roi["x"] + 5, roi["y"] + 2, roi["x"] + 70, roi["y"] + 26),
            radius=12,
            fill=(94, 68, 42),
        )
        draw.ellipse((roi["x"] + 13, roi["y"] + 6, roi["x"] + 31, roi["y"] + 19), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 20, roi["y"] + 10, roi["x"] + 25, roi["y"] + 15), fill=(94, 68, 42))
        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 17)
        draw.text((roi["x"] + 39, roi["y"] + 5), "16", fill=(255, 255, 255), font=font)
        ocr_item = {
            "text": "16",
            "rect": {"x": roi["x"] + 36, "y": roi["y"] + 1, "width": 23, "height": 18},
        }

        it = feed_eye._refine_overlay_item_with_visual_digit(img, roi, ocr_item)

        self.assertEqual(it["text"], "16")
        self.assertIn("overlay_anchor_center_y", it)
        self.assertIn("overlay_visual_score", it)
        self.assertIn("overlay_erase_rect", it)
        self.assertGreaterEqual(it["rect"]["y"], roi["y"] + 4)
        self.assertLess(it["rect"]["height"], ocr_item["rect"]["height"])

    def test_visual_fallback_ignores_bright_decoration_above_digit_lane(self):
        title = {"text": "本地人私藏不踩雷", "rect": {"x": 286, "y": 1005, "width": 216, "height": 48}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(74, 88, 96),
        )
        draw.rounded_rectangle(
            (roi["x"] + 8, roi["y"] + 8, roi["x"] + 70, roi["y"] + 31),
            radius=12,
            fill=(128, 138, 144),
        )
        draw.ellipse((roi["x"] + 14, roi["y"] + 14, roi["x"] + 31, roi["y"] + 24), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 20, roi["y"] + 17, roi["x"] + 25, roi["y"] + 22), fill=(128, 138, 144))
        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 17)
        draw.text((roi["x"] + 37, roi["y"] + 9), "16", fill=(255, 255, 255), font=font)
        draw.rectangle((roi["x"] + 58, roi["y"] + 2, roi["x"] + 70, roi["y"] + 12), fill=(248, 248, 248))

        it = feed_eye._visual_overlay_item_from_roi(img, roi)

        self.assertIsNotNone(it)
        self.assertEqual(it["text"], "16")
        self.assertLess(it["rect"]["height"], 18)
        self.assertLess(it["rect"]["width"], 24)

    def test_refines_two_digit_ocr_box_that_includes_eye_icon(self):
        title = {"text": "我没办法了。。失眠。失", "rect": {"x": 286, "y": 660, "width": 220, "height": 21}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(245, 245, 245),
        )
        draw.rounded_rectangle(
            (roi["x"] + 6, roi["y"] + 8, roi["x"] + 62, roi["y"] + 31),
            radius=12,
            fill=(176, 176, 176),
        )
        draw.ellipse((roi["x"] + 14, roi["y"] + 14, roi["x"] + 31, roi["y"] + 23), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 20, roi["y"] + 16, roi["x"] + 25, roi["y"] + 21), fill=(176, 176, 176))
        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 17)
        draw.text((roi["x"] + 40, roi["y"] + 8), "18", fill=(255, 255, 255), font=font)
        merged = {
            "text": "18",
            "rect": {"x": roi["x"] + 9, "y": roi["y"] + 7, "width": 39, "height": 20},
        }

        it = feed_eye._refine_overlay_item_with_visual_digit(img, roi, merged)

        self.assertEqual(it["text"], "18")
        self.assertIn("overlay_anchor_center_y", it)
        self.assertNotIn("overlay_erase_rect", it)
        self.assertLessEqual(it.get("overlay_left_nudge_px", 0), 0)
        self.assertGreater(it["rect"]["x"], roi["x"] + 32)
        self.assertLess(it["rect"]["width"], merged["rect"]["width"])

    def test_ignores_tiny_product_text_in_overlay_roi_and_uses_visual_digit(self):
        title = {"text": "终于找到专为儿童设计的", "rect": {"x": 19, "y": 681, "width": 225, "height": 22}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (thumb["x"], thumb["y"], thumb["x"] + thumb["width"], thumb["y"] + thumb["height"]),
            fill=(110, 130, 112),
        )
        draw.rounded_rectangle(
            (roi["x"] + 7, roi["y"] + 3, roi["x"] + 56, roi["y"] + 26),
            radius=12,
            fill=(92, 104, 94),
        )
        draw.ellipse((roi["x"] + 14, roi["y"] + 10, roi["x"] + 30, roi["y"] + 19), fill=(248, 248, 248))
        draw.ellipse((roi["x"] + 20, roi["y"] + 12, roi["x"] + 24, roi["y"] + 17), fill=(92, 104, 94))
        draw.line((roi["x"] + 41, roi["y"] + 7, roi["x"] + 41, roi["y"] + 20), fill=(255, 255, 255), width=2)
        tiny_product_text = {
            "text": "S41",
            "rect": {"x": roi["x"] + 100, "y": roi["y"] + 9, "width": 15, "height": 6},
        }

        with patch("data_polisher.feed_eye.cli.detect_items_with_paddle", return_value=[]):
            it = feed_eye.find_card_eye_number_item(img, [title, tiny_product_text], title)

        self.assertEqual(it["text"], "1")
        self.assertLess(it["rect"]["x"], roi["x"] + 55)

    def test_mixed_cover_text_candidate_defers_to_visual_digits(self):
        title = {"text": "暑假不用去远方，彭州就", "rect": {"x": 288, "y": 655, "width": 224, "height": 21}}
        thumb = feed_eye._infer_thumbnail_rect(title, 540, 1200)
        roi = feed_eye._overlay_strip_roi(thumb)
        img = Image.new("RGB", (540, 1200), color=(245, 245, 245))
        mixed_cover_text = {
            "text": "7大叶",
            "rect": {"x": roi["x"] + 51, "y": roi["y"] + 7, "width": 30, "height": 11},
        }
        visual = {
            "text": "13",
            "rect": {"x": roi["x"] + 37, "y": roi["y"] + 4, "width": 15, "height": 13},
            "overlay_visual_score": 0.48,
            "overlay_anchor_center_y": roi["y"] + 11.5,
        }

        with patch("data_polisher.feed_eye._visual_overlay_item_from_roi", return_value=visual):
            it = feed_eye._refine_overlay_item_with_visual_digit(img, roi, mixed_cover_text)

        self.assertEqual(it["text"], "13")
        self.assertEqual(it["rect"], visual["rect"])

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
