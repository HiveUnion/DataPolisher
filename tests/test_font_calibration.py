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

        self.assertGreaterEqual(zero_bbox[2] - zero_bbox[0], 28)
        self.assertGreaterEqual(one_bbox[2] - one_bbox[0], 16)

    def test_body_native_font_uses_small_visual_adjustment(self):
        self.assertEqual(cli.BODY_NATIVE_FONT_SIZE_ADJUST, -1)

    def test_body_native_font_uses_consistent_bold_variant(self):
        self.assertEqual(cli.BODY_NATIVE_FORCE_EDGE_VARIANT, "w2:quantized")
        self.assertEqual(cli.BODY_NATIVE_FORCE_ALPHA_STRENGTH, 0.75)

    def test_body_font_matching_prefers_closer_digit_widths(self):
        source_stats = {
            "target_height": 41,
            "target_density": 0.44,
            "char_widths": {
                "0": [29],
                "1": [18],
                "6": [29],
                "%": [46],
            },
        }

        chosen = cli.choose_best_body_font(["1000", "300"], source_stats)

        self.assertEqual(chosen["font_path"], "/System/Library/Fonts/SFNS.ttf")

    def test_body_uses_row_atlas_when_it_covers_all_replacements(self):
        atlas = {"glyphs": {char: object() for char in "013.%"}}

        self.assertTrue(cli.should_use_body_row_atlas(atlas, ["1000", "300", "30.0%"]))

    def test_body_uses_native_font_when_row_atlas_is_missing_replacement_chars(self):
        atlas = {"glyphs": {char: object() for char in "0169"}}

        self.assertFalse(cli.should_use_body_row_atlas(atlas, ["1000", "300", "30.0%"]))


if __name__ == "__main__":
    unittest.main()
