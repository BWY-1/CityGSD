import os
import unittest

from utils.camera_utils import auxiliary_image_path


class AuxiliaryImagePathTests(unittest.TestCase):
    def test_images_layout(self):
        self.assertEqual(
            auxiliary_image_path("data/CameraA/train/images/a.jpg", "depths"),
            os.path.join("data", "CameraA", "train", "depths", "a.png"),
        )

    def test_rgbs_layout(self):
        self.assertEqual(
            auxiliary_image_path("data/scene/train/rgbs/a.JPG", "mask"),
            os.path.join("data", "scene", "train", "mask", "a.png"),
        )

    def test_unknown_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            auxiliary_image_path("data/train/a.jpg", "depths")


if __name__ == "__main__":
    unittest.main()
