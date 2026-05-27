import unittest

from PIL import Image

from data_polisher import cli


class FontCalibrationTests(unittest.TestCase):
    def test_choose_font_size_matches_target_rendered_height(self):
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        target_height = 41
        texts = ["1000", "300", "30.0%"]

        size = cli.choose_font_size_for_rendered_height(font_path, texts, target_height)
        font = cli.load_font_by_path(font_path, size)
        self.assertIsNotNone(font)

        heights = []
        for text in texts:
            bbox = cli.rendered_ink_bbox(text, font)
            heights.append(bbox[3] - bbox[1])

        median_height = sorted(heights)[len(heights) // 2]
        self.assertLessEqual(abs(median_height - target_height), 2)

    def test_row_atlas_synthesizes_missing_metric_glyphs(self):
        glyph = Image.new("RGBA", (12, 20), (0, 0, 0, 255))
        atlas = {
            "glyphs": {
                "0": {
                    "image": glyph,
                    "height": 20,
                    "width": 12,
                    "row_height": 20,
                    "row_y_offset": 0,
                }
            },
            "reference_height": 20,
            "glyph_spacing": 2,
        }
        image = Image.new("RGB", (100, 60), "white")
        result = cli.compose_text_from_row_atlas(
            image,
            atlas,
            {"x": 10, "y": 10, "width": 30, "height": 20},
            "30",
        )

        self.assertIsNotNone(result)

    def test_body_native_font_matches_wide_source_digits(self):
        font = cli.load_font_by_path(cli.BODY_NATIVE_FONT_PATH, 55)
        self.assertIsNotNone(font)

        zero_bbox = cli.rendered_ink_bbox("0", font)
        one_bbox = cli.rendered_ink_bbox("1", font)

        self.assertGreaterEqual(zero_bbox[2] - zero_bbox[0], 8)
        self.assertGreaterEqual(one_bbox[2] - one_bbox[0], 4)

    def test_body_native_font_uses_small_visual_adjustment(self):
        self.assertEqual(cli.BODY_NATIVE_FONT_SIZE_ADJUST, 2)

    def test_body_native_font_uses_slight_bold_variant(self):
        self.assertEqual(cli.BODY_NATIVE_FORCE_EDGE_VARIANT, "w1x:quantized")
        self.assertEqual(cli.BODY_NATIVE_FORCE_ALPHA_STRENGTH, 0.25)

    def test_feed_overlay_forced_font_prefers_calibrated_overlay_digits(self):
        chosen = cli.red_number_forced_font_for_standalone_patch(
            ink_rect={"x": 0, "y": 0, "width": 18, "height": 12},
            original_text="40",
            new_text="94",
            overlay_views_ink=True,
        )

        if cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT.is_file():
            self.assertEqual(str(chosen["font_path"]), str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT))
            self.assertTrue(str(chosen["font_path"]).endswith("FZYouHS-508R.ttf"))
            expected_size = cli.choose_feed_overlay_font_size(
                str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT),
                ["40"],
                {"x": 0, "y": 0, "width": 18, "height": 12},
            )
            self.assertEqual(chosen["font_size"], expected_size)
            self.assertLessEqual(chosen["font_size"], cli.FEED_OVERLAY_VIEWS_FONT_SIZE)
            self.assertNotIn("overlay_direct_mask", chosen)
            self.assertNotIn("overlay_scale_x", chosen)
            self.assertEqual(chosen["overlay_visual_dy"], cli._feed_overlay_visual_dy("40", "94"))
            self.assertEqual(chosen["overlay_alpha_gamma"], cli.FEED_OVERLAY_VIEWS_ALPHA_GAMMA)
            font = cli.load_font_by_path(str(chosen["font_path"]), chosen["font_size"])
            bbox = cli.rendered_ink_bbox("40", font)
            self.assertLessEqual(bbox[3] - bbox[1], 12)
            self.assertLessEqual(bbox[2] - bbox[0], 18)

    def test_feed_overlay_antialias_params_adapt_to_sharp_source_digits(self):
        soft_style = {
            "density": 0.253,
            "edge_ratio": 0.30,
            "alpha_summary": {"p10": 180, "p50": 246, "p90": 255},
        }
        sharp_style = {
            "density": 0.138,
            "edge_ratio": 0.19,
            "alpha_summary": {"p10": 211, "p50": 252, "p90": 255},
        }

        soft_mode, soft_threshold, soft_strength = cli.feed_overlay_antialias_params(soft_style)
        sharp_mode, sharp_threshold, sharp_strength = cli.feed_overlay_antialias_params(sharp_style)

        self.assertEqual(soft_mode, "aa")
        self.assertEqual(soft_threshold, cli.FEED_OVERLAY_AA_TRIM_THRESHOLD)
        self.assertEqual(soft_strength, cli.FEED_OVERLAY_AA_BLEND_STRENGTH)
        self.assertEqual(sharp_mode, "quantized")
        self.assertEqual(sharp_threshold, cli.FEED_OVERLAY_SHARP_QUANT_STEP)
        self.assertEqual(sharp_strength, cli.FEED_OVERLAY_SHARP_QUANT_GAIN)

    def test_feed_overlay_calibrates_font_size_by_recreating_source_digits(self):
        if not cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT.is_file():
            self.skipTest("bundled feed overlay font missing")

        font14 = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 14)
        background = Image.new("RGB", (100, 60), (144, 168, 167))
        origin = (24, 20)
        base_mask = cli.text_mask_for_candidate(
            background.size,
            "16",
            font14,
            origin,
            font_path_hint=str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT),
        )
        source_mask = cli.feed_overlay_mask_from_variant(
            base_mask,
            {"alpha_values": [255]},
            "quantized64",
            cli.FEED_OVERLAY_SHARP_QUANT_GAIN,
        )
        source = cli.composite_text_mask(background, source_mask, (255, 255, 255))
        bbox = source_mask.getbbox()
        rect = {"x": bbox[0], "y": bbox[1], "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]}
        forced = {
            "font_size": 13,
            "font_path": str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT),
            "font_match": {"synthetic_initial_estimate": True},
            "overlay_visual_dx": 0,
            "overlay_visual_dy": 0,
            "overlay_alpha_gain": 1.0,
            "overlay_alpha_gamma": 1.0,
            "overlay_color": (255, 255, 255),
        }

        _image, report = cli.patch_ocr_rect_with_glyphs(
            source.copy(),
            source,
            rect,
            "38",
            None,
            "16",
            raw_text="16",
            forced_font=forced,
            overlay_use_input_rect=True,
            overlay_erase_rect=rect,
            overlay_views_ink=True,
        )

        calibration = report["calibration"]
        self.assertEqual(calibration["font_size"], 14)
        self.assertEqual(calibration["overlay_fit_font_size_adjust"], (13, 14))
        self.assertLess(calibration["overlay_origin_rmse"], 12)
        self.assertEqual(calibration["new_edge_variant"], calibration["overlay_origin_edge_variant"])
        self.assertEqual(
            calibration["new_alpha_match_strength"],
            calibration["overlay_origin_alpha_match_strength"],
        )

    def test_feed_overlay_reuses_origin_edge_style_and_left_bearing(self):
        if not cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT.is_file():
            self.skipTest("bundled feed overlay font missing")

        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 13)
        background = Image.new("RGB", (100, 60), (42, 44, 46))
        origin = (24, 20)
        base_mask = cli.text_mask_for_candidate(
            background.size,
            "16",
            font,
            origin,
            font_path_hint=str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT),
        )
        source_mask = cli.adjust_mask_alpha(base_mask, gain=0.86, gamma=0.76)
        source = cli.composite_text_mask(background, source_mask, (254, 254, 252))
        bbox = source_mask.getbbox()
        rect = {"x": bbox[0], "y": bbox[1], "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]}

        _image, report = cli.patch_ocr_rect_with_glyphs(
            source.copy(),
            source,
            rect,
            "46",
            None,
            "16",
            raw_text="16",
            overlay_use_input_rect=True,
            overlay_erase_rect=rect,
            overlay_views_ink=True,
        )

        calibration = report["calibration"]
        cal_font = cli.load_font_by_path(calibration["font_path"], calibration["font_size"])
        new_bbox = cli.rendered_ink_bbox("46", cal_font)
        self.assertEqual(
            calibration["overlay_new_font_origin"][0],
            calibration["overlay_font_origin"][0] + calibration["overlay_origin_bbox"][0] - new_bbox[0],
        )
        self.assertEqual(calibration["new_edge_variant"], calibration["overlay_origin_edge_variant"])
        self.assertEqual(
            calibration["new_alpha_match_strength"],
            calibration["overlay_origin_alpha_match_strength"],
        )

    def test_feed_overlay_stroke_erase_removes_antialias_halo(self):
        if not cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT.is_file():
            self.skipTest("bundled feed overlay font missing")

        import numpy as np

        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 13)
        image = Image.new("RGB", (100, 60), (45, 47, 49))
        mask = cli.text_mask_for_candidate(
            image.size,
            "16",
            font,
            (24, 20),
            font_path_hint=str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT),
        )
        source = cli.composite_text_mask(image, mask, (255, 255, 255))
        bbox = mask.getbbox()
        rect = {"x": bbox[0], "y": bbox[1], "width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]}

        clean = cli.inpaint_overlay_views_stroke_fill(
            source.copy(),
            source,
            cli.expand_rect(rect, 3, {"width": source.width, "height": source.height}),
        )

        old_stroke = np.asarray(mask, dtype=np.uint8) > 0
        before = np.asarray(source.convert("L"), dtype=np.float32)[old_stroke].mean()
        after = np.asarray(clean.convert("L"), dtype=np.float32)[old_stroke].mean()
        self.assertLess(after, before - 90)

    def test_feed_overlay_stroke_erase_preserves_top_right_capsule_highlight(self):
        if not cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT.is_file():
            self.skipTest("bundled feed overlay font missing")

        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 13)
        image = Image.new("RGB", (100, 60), (156, 154, 82))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((15, 20, 65, 43), radius=12, fill=(150, 148, 80))
        draw.text((35, 25), "17", font=font, fill=(255, 255, 246))
        draw.rectangle((66, 21, 68, 23), fill=(244, 244, 218))

        rect = {"x": 32, "y": 22, "width": 34, "height": 22}
        clean = cli.inpaint_overlay_views_stroke_fill(image.copy(), image, rect)

        self.assertEqual(clean.getpixel((67, 22)), image.getpixel((67, 22)))

    def test_feed_overlay_stroke_erase_preserves_bottom_right_capsule_edge(self):
        if not cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT.is_file():
            self.skipTest("bundled feed overlay font missing")

        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), 13)
        image = Image.new("RGB", (100, 60), (236, 238, 240))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((15, 20, 66, 43), radius=12, fill=(176, 178, 180))
        draw.text((35, 25), "15", font=font, fill=(255, 255, 255))
        draw.point((67, 43), fill=(218, 220, 222))

        rect = {"x": 32, "y": 22, "width": 34, "height": 22}
        clean = cli.inpaint_overlay_views_stroke_fill(image.copy(), image, rect)

        self.assertEqual(clean.getpixel((67, 43)), image.getpixel((67, 43)))

    def test_feed_overlay_font_size_handles_loose_ocr_boxes(self):
        if not cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT.is_file():
            self.skipTest("bundled feed overlay font missing")

        size = cli.choose_feed_overlay_font_size(
            str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT),
            ["22", "62"],
            {"x": 0, "y": 0, "width": 23, "height": 17},
        )
        font = cli.load_font_by_path(str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT), size)
        bbox = cli.rendered_ink_bbox("62", font)

        self.assertEqual(size, 13)
        self.assertLessEqual(bbox[3] - bbox[1], 11)

    def test_feed_overlay_font_size_does_not_overfit_narrow_ones(self):
        if not cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT.is_file():
            self.skipTest("bundled feed overlay font missing")

        size = cli.choose_feed_overlay_font_size(
            str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT),
            ["11", "67"],
            {"x": 0, "y": 0, "width": 12, "height": 12},
        )

        self.assertEqual(size, 13)

    def test_body_font_matching_prefers_closer_digit_widths(self):
        from data_polisher.red_number_fonts import RED_NUMBER_BOLD

        source_stats = {
            "target_height": 41,
            "target_density": 0.44,
            "target_edge_ratio": 0.15,
            "char_widths": {
                "0": [29],
                "1": [18],
                "6": [29],
                "%": [46],
            },
        }

        chosen = cli.choose_best_body_font(["1000", "300"], source_stats)

        if RED_NUMBER_BOLD.is_file():
            self.assertTrue(str(chosen["font_path"]).endswith("REDNumber-Bold.otf"))
            self.assertIsNotNone(chosen["score"])
        else:
            self.assertIn(chosen["font_path"], cli.BODY_NATIVE_FONT_CANDIDATE_PATHS)
            self.assertIsNotNone(chosen["score"])

    def test_body_uses_row_atlas_when_it_covers_all_replacements(self):
        atlas = {"glyphs": {char: object() for char in "013.%"}}

        self.assertTrue(cli.should_use_body_row_atlas(atlas, ["1000", "300", "30.0%"]))

    def test_body_uses_native_font_when_row_atlas_is_missing_replacement_chars(self):
        atlas = {"glyphs": {char: object() for char in "0169"}}

        self.assertFalse(cli.should_use_body_row_atlas(atlas, ["1000", "300", "30.0%"]))

    def test_style_distance_prefers_balanced_weight_over_too_light_or_too_dark(self):
        target = {
            "density": 0.466,
            "edge_ratio": 0.198,
            "alpha_summary": {"p10": 3, "p50": 230, "p90": 255},
        }
        lighter = {
            "density": 0.450,
            "edge_ratio": 0.475,
            "alpha_summary": {"p10": 22, "p50": 255, "p90": 255},
        }
        balanced = {
            "density": 0.529,
            "edge_ratio": 0.380,
            "alpha_summary": {"p10": 20, "p50": 255, "p90": 255},
        }
        darker = {
            "density": 0.768,
            "edge_ratio": 0.219,
            "alpha_summary": {"p10": 35, "p50": 255, "p90": 255},
        }

        self.assertLess(cli.style_distance(target, balanced), cli.style_distance(target, lighter))
        self.assertLess(cli.style_distance(target, balanced), cli.style_distance(target, darker))

    def test_overlay_views_left_nudge_negative_when_replacement_wider(self):
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        font = cli.load_font_by_path(font_path, 18)
        if font is None:
            self.skipTest("Arial Bold not available")
        self.assertEqual(cli._overlay_views_left_nudge_px("40", "40", font=font), 0)
        n = cli._overlay_views_left_nudge_px("40", "999", font=font)
        self.assertLess(n, 0)
        self.assertGreaterEqual(n, -10)


if __name__ == "__main__":
    unittest.main()
