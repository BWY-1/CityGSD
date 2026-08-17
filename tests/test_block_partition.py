import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "scene", "block_partition.py"
)
SPEC = importlib.util.spec_from_file_location("block_partition_under_test", MODULE_PATH)
block_partition = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = block_partition
SPEC.loader.exec_module(block_partition)


class BlockPartitionTests(unittest.TestCase):
    def make_manifest(self, folder):
        os.makedirs(os.path.join(folder, "block_000_000"))
        np.save(os.path.join(folder, "block_000_000", "point_indices.npy"), [0, 2])
        manifest = {
            "version": 1,
            "blocks": [{
                "id": "000_000",
                "core_aabb": [[0, 0, 0], [1, 1, 1]],
                "train_aabb": [[-0.1, -0.1, 0], [1.1, 1.1, 1]],
                "core_max_inclusive": [False, True, True],
                "train_camera_names": ["a", "b"],
                "test_camera_names": ["c"],
                "point_indices": "block_000_000/point_indices.npy",
            }],
        }
        path = os.path.join(folder, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        return path

    def test_manifest_and_half_open_core_ownership(self):
        with tempfile.TemporaryDirectory() as folder:
            spec = block_partition.load_block_spec(self.make_manifest(folder), "000_000")
            points = np.asarray([[0, 0, 0], [0.999, 1, 1], [1, 0.5, 0.5]])
            np.testing.assert_array_equal(spec.contains_core_points(points), [True, True, False])
            np.testing.assert_array_equal(spec.contains_train_points(points), [True, True, True])

    def test_camera_and_point_cloud_filtering(self):
        Camera = type("Camera", (), {})
        cameras = []
        for name in ("a", "b", "c"):
            camera = Camera()
            camera.image_name = name
            cameras.append(camera)
        filtered = block_partition.filter_camera_infos(cameras, {"b", "c"})
        self.assertEqual([camera.image_name for camera in filtered], ["b", "c"])

        cloud = block_partition.BasicPointCloud(
            points=np.arange(9).reshape(3, 3),
            colors=np.ones((3, 3)),
            normals=np.zeros((3, 3)),
        )
        selected = block_partition.filter_basic_point_cloud(cloud, [0, 2])
        np.testing.assert_array_equal(selected.points, cloud.points[[0, 2]])

    def test_unknown_block_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.make_manifest(folder)
            with self.assertRaises(ValueError):
                block_partition.load_block_spec(path, "missing")


if __name__ == "__main__":
    unittest.main()
