import unittest

import numpy as np

from eval_f1 import summarize_distances


class GeometryMetricTests(unittest.TestCase):
    def test_summary_uses_requested_threshold(self):
        metrics = summarize_distances([0.1, 0.6], [0.2, 0.4], threshold=0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["fscore"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["chamfer_l1"], 0.325)

    def test_empty_geometry_is_safe(self):
        metrics = summarize_distances([], [0.1], threshold=0.5)
        self.assertEqual(metrics["fscore"], 0.0)


if __name__ == "__main__":
    unittest.main()
