import unittest
from unittest.mock import patch

from data_polisher import ocr


class OcrFallbackTests(unittest.TestCase):
    def test_empty_apple_items_fall_back_to_paddle(self):
        class FakeEngine:
            def predict(self, image):
                return [{"rec_texts": ["耳机"], "rec_boxes": [[10, 20, 30, 40]]}]

        with patch("data_polisher.ocr._apple_vision_available", return_value=True), patch(
            "data_polisher.ocr_apple.detect_items", return_value=[]
        ), patch("data_polisher.ocr.get_ocr_engine", return_value=FakeEngine()):
            items = ocr.detect_items_with_paddle(object())

        self.assertEqual(items[0]["text"], "耳机")
        self.assertEqual(items[0]["rect"], {"x": 10, "y": 20, "width": 20, "height": 20})

    def test_empty_apple_bounds_fall_back_to_paddle(self):
        class FakeEngine:
            def predict(self, image):
                return [{"rec_texts": ["1"], "rec_boxes": [[10, 20, 30, 40]]}]

        with patch("data_polisher.ocr._apple_vision_available", return_value=True), patch(
            "data_polisher.ocr_apple.detect_bounds", return_value=None
        ), patch("data_polisher.ocr.get_ocr_engine", return_value=FakeEngine()):
            bounds = ocr.detect_bounds_with_paddle(object())

        self.assertEqual(bounds, {"x": 10, "y": 20, "width": 20, "height": 20})


if __name__ == "__main__":
    unittest.main()
