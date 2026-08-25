import tempfile
import unittest

import torch

from local_update.local_optimizer import capture_static_rows, restore_static_rows
from local_update.roi import ROISelector, parse_bbox


class LocalUpdateTests(unittest.TestCase):
    def test_roi_boundary_and_observable_intersection(self):
        points = torch.tensor([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [3.0, 0.0, 0.0]])
        selector = ROISelector(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0]), 1.0)
        core, boundary, roi = selector.build_masks(points)
        observable = torch.tensor([True, False, True])
        self.assertEqual(core.tolist(), [True, False, False])
        self.assertEqual(boundary.tolist(), [False, True, False])
        self.assertEqual((roi & observable).tolist(), [True, False, False])

    def test_bbox_file_parsing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            handle.write('{"bbox": [0, 1, 2, 3, 4, 5]}')
            handle.flush()
            lower, upper = parse_bbox("", handle.name)
        self.assertEqual(lower.tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(upper.tolist(), [3.0, 4.0, 5.0])

    def test_static_rows_are_bitwise_restored_after_adam_momentum(self):
        parameter = torch.nn.Parameter(torch.tensor([[1.0], [2.0], [3.0]]))
        optimizer = torch.optim.Adam([{"params": [parameter], "name": "anchor"}], lr=0.1)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        update_mask = torch.tensor([False, True, False])
        before = parameter.detach().clone()
        snapshot = capture_static_rows(optimizer.param_groups, update_mask)
        parameter.grad = torch.zeros_like(parameter)
        parameter.grad[update_mask] = 1.0
        optimizer.step()  # Momentum would move the static rows without restoration.
        drift_before_restore = restore_static_rows(optimizer.param_groups, snapshot)

        self.assertGreater(drift_before_restore, 0.0)
        self.assertTrue(torch.equal(parameter.detach()[~update_mask], before[~update_mask]))
        self.assertFalse(torch.equal(parameter.detach()[update_mask], before[update_mask]))


if __name__ == "__main__":
    unittest.main()
