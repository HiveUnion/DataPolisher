import unittest

from data_polisher.gui import _parse_int_range_text, _parse_range_pair


class GuiRangeParseTests(unittest.TestCase):
    def test_pair_basic(self):
        self.assertEqual(_parse_range_pair("10", "20", "x", lower_floor=0), (10, 20))

    def test_pair_swap(self):
        self.assertEqual(_parse_range_pair("50", "30", "x", lower_floor=0), (30, 50))

    def test_pair_empty(self):
        with self.assertRaises(ValueError):
            _parse_range_pair("", "5", "x", lower_floor=0)

    def test_single_number(self):
        self.assertEqual(_parse_int_range_text("300", "x", lower_floor=0), (300, 300))

    def test_range_ordered(self):
        self.assertEqual(_parse_int_range_text("10-20", "x", lower_floor=0), (10, 20))

    def test_range_reversed_swapped(self):
        self.assertEqual(_parse_int_range_text("50-30", "x", lower_floor=0), (30, 50))

    def test_en_dash_normalized(self):
        self.assertEqual(_parse_int_range_text("80–120", "x", lower_floor=0), (80, 120))

    def test_exposure_floor(self):
        self.assertEqual(_parse_int_range_text("1-5", "曝光", lower_floor=1), (1, 5))
        with self.assertRaises(ValueError):
            _parse_int_range_text("0-3", "曝光", lower_floor=1)


if __name__ == "__main__":
    unittest.main()
