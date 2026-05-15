import unittest

from data_polisher.core import calculate_metrics, format_percent
from data_polisher.cli import metric_text_changed


class MetricsFormatTests(unittest.TestCase):
    def test_percent_drops_trailing_dot_zero(self):
        self.assertEqual(format_percent(0.3), "30%")
        self.assertEqual(format_percent(0.0), "0%")

    def test_percent_keeps_meaningful_one_decimal(self):
        self.assertEqual(format_percent(0.074), "7.4%")
        self.assertEqual(format_percent(0.182), "18.2%")

    def test_calculate_metrics_uses_compact_percent_format(self):
        metrics = calculate_metrics(
            exposure=1000,
            views=300,
            likes=0,
            comments=0,
            collects=0,
            shares=0,
        )

        self.assertEqual(metrics["click_rate_text"], "30%")
        self.assertEqual(metrics["interaction_rate_text"], "0%")
        self.assertEqual(metrics["views_text"], "300")
        self.assertEqual(metrics["header_views_text"], metrics["views_text"])

    def test_metric_text_changed_skips_identical_values(self):
        self.assertFalse(metric_text_changed("122", "122"))
        self.assertFalse(metric_text_changed("36", "36"))
        self.assertFalse(metric_text_changed("0%", "0%"))
        self.assertTrue(metric_text_changed("18.2%", "29.5%"))


if __name__ == "__main__":
    unittest.main()
