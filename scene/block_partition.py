import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from utils.graphics_utils import BasicPointCloud


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    core_min: np.ndarray
    core_max: np.ndarray
    train_min: np.ndarray
    train_max: np.ndarray
    train_camera_names: frozenset
    test_camera_names: frozenset
    point_indices_path: str
    manifest_hash: str
    core_max_inclusive: np.ndarray

    def contains_train_points(self, points):
        points = np.asarray(points)
        return np.logical_and(points >= self.train_min, points <= self.train_max).all(axis=-1)

    def contains_core_points(self, points):
        points = np.asarray(points)
        lower = points >= self.core_min
        upper = np.where(self.core_max_inclusive, points <= self.core_max, points < self.core_max)
        return np.logical_and(lower, upper).all(axis=-1)


def _manifest_digest(data):
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_block_spec(manifest_path, block_id):
    manifest_path = os.path.abspath(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported block manifest version: {manifest.get('version')}")

    matches = [block for block in manifest["blocks"] if block["id"] == block_id]
    if len(matches) != 1:
        raise ValueError(f"Block {block_id!r} was not found exactly once in {manifest_path}")
    block = matches[0]
    manifest_dir = os.path.dirname(manifest_path)
    point_indices_path = block["point_indices"]
    if not os.path.isabs(point_indices_path):
        point_indices_path = os.path.join(manifest_dir, point_indices_path)

    return BlockSpec(
        block_id=block_id,
        core_min=np.asarray(block["core_aabb"][0], dtype=np.float32),
        core_max=np.asarray(block["core_aabb"][1], dtype=np.float32),
        train_min=np.asarray(block["train_aabb"][0], dtype=np.float32),
        train_max=np.asarray(block["train_aabb"][1], dtype=np.float32),
        train_camera_names=frozenset(block["train_camera_names"]),
        test_camera_names=frozenset(block.get("test_camera_names", [])),
        point_indices_path=os.path.abspath(point_indices_path),
        manifest_hash=_manifest_digest(manifest),
        core_max_inclusive=np.asarray(block.get("core_max_inclusive", [False, False, True]), dtype=bool),
    )


def filter_camera_infos(cameras, allowed_names):
    allowed_names = set(allowed_names)
    return [camera for camera in cameras if camera.image_name in allowed_names]


def filter_basic_point_cloud(point_cloud, indices):
    if point_cloud is None:
        raise ValueError("A sparse point cloud is required for spatial block training")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Point indices must be a one-dimensional array")
    if indices.size == 0:
        raise ValueError("The selected spatial block contains no sparse points")
    if indices.min() < 0 or indices.max() >= len(point_cloud.points):
        raise IndexError("Block point indices are outside the source point cloud")
    return BasicPointCloud(
        points=np.asarray(point_cloud.points)[indices],
        colors=np.asarray(point_cloud.colors)[indices],
        normals=np.asarray(point_cloud.normals)[indices],
    )
