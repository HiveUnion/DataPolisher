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
            self.assertTrue(str(chosen["font_path"]).endswith("jlm_cmss10.ttf"))
            expected_size = cli.choose_feed_overlay_font_size(
                str(cli.BUNDLED_FEED_OVERLAY_VIEWS_FONT),
                ["40", "94"],
                {"x": 0, "y": 0, "width": 18, "height": 12},
            )
            self.assertEqual(chosen["font_size"], expected_size)
            self.assertEqual(chosen["font_size"], 17)
            self.assertEqual(chosen["overlay_visual_dx"], cli.FEED_OVERLAY_VIEWS_DX)
            self.assertEqual(chosen["overlay_visual_dy"], cli.FEED_OVERLAY_VIEWS_DY)
            self.assertEqual(chosen["overlay_alpha_gamma"], cli.FEED_OVERLAY_VIEWS_ALPHA_GAMMA)
            font = cli.load_font_by_path(str(chosen["font_path"]), chosen["font_size"])
            bbox = cli.rendered_ink_bbox("90", font)
            self.assertLessEqual(bbox[3] - bbox[1], 12)
            self.assertLessEqual(bbox[2] - bbox[0], 18)

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
