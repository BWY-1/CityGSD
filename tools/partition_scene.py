#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scene.colmap_loader import (  # noqa: E402
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
    qvec2rotmat,
)
from utils.graphics_utils import focal2fov  # noqa: E402


def read_colmap_model(source_path):
    sparse = os.path.join(source_path, "sparse", "0")
    try:
        extrinsics = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
        intrinsics = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(os.path.join(sparse, "images.txt"))
        intrinsics = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
    try:
        points, _, _ = read_points3D_binary(os.path.join(sparse, "points3D.bin"))
    except Exception:
        points, _, _ = read_points3D_text(os.path.join(sparse, "points3D.txt"))
    return extrinsics, intrinsics, np.asarray(points, dtype=np.float64)


def camera_footprint(extrinsic, intrinsic, reference_z):
    rotation = qvec2rotmat(extrinsic.qvec).T
    center = np.linalg.inv(
        np.block([[rotation.T, np.asarray(extrinsic.tvec)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    )[:3, 3]
    if intrinsic.model == "SIMPLE_PINHOLE":
        fx = fy = intrinsic.params[0]
    elif intrinsic.model in ("PINHOLE", "OPENCV"):
        fx, fy = intrinsic.params[:2]
    else:
        raise ValueError(f"Unsupported camera model {intrinsic.model}")
    fov_x = focal2fov(fx, intrinsic.width)
    fov_y = focal2fov(fy, intrinsic.height)
    corners = []
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        local_ray = np.asarray([sx * math.tan(fov_x / 2), sy * math.tan(fov_y / 2), 1.0])
        world_ray = rotation @ local_ray
        if abs(world_ray[2]) < 1e-8:
            continue
        distance = (reference_z - center[2]) / world_ray[2]
        if distance > 0:
            corners.append((center + distance * world_ray)[:2])
    if not corners:
        return center[:2], center[:2]
    corners = np.asarray(corners)
    return corners.min(axis=0), corners.max(axis=0)


def intersects_2d(a_min, a_max, b_min, b_max):
    return bool(np.logical_and(a_max >= b_min, b_max >= a_min).all())


def main():
    parser = argparse.ArgumentParser(description="Partition a COLMAP scene into overlapping XY blocks")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--grid", nargs=2, type=int, default=[3, 3], metavar=("NX", "NY"))
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--min_cameras", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=0.005)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.grid) < 1 or not 0 <= args.overlap < 0.5:
        parser.error("grid dimensions must be positive and overlap must be in [0, 0.5)")

    extrinsics, intrinsics, points = read_colmap_model(args.source_path)
    q = args.quantile
    xy_min = np.quantile(points[:, :2], q, axis=0)
    xy_max = np.quantile(points[:, :2], 1 - q, axis=0)
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    reference_z = float(np.median(points[:, 2]))
    nx, ny = args.grid
    step = (xy_max - xy_min) / np.asarray([nx, ny])

    footprints = {}
    for extrinsic in extrinsics.values():
        image_name = os.path.splitext(os.path.basename(extrinsic.name))[0]
        footprints[image_name] = camera_footprint(
            extrinsic, intrinsics[extrinsic.camera_id], reference_z
        )

    os.makedirs(args.output, exist_ok=True)
    blocks = []
    for ix in range(nx):
        for iy in range(ny):
            core_xy_min = xy_min + step * np.asarray([ix, iy])
            core_xy_max = xy_min + step * np.asarray([ix + 1, iy + 1])
            train_xy_min = np.maximum(xy_min, core_xy_min - args.overlap * step)
            train_xy_max = np.minimum(xy_max, core_xy_max + args.overlap * step)
            train_min = np.asarray([train_xy_min[0], train_xy_min[1], z_min])
            train_max = np.asarray([train_xy_max[0], train_xy_max[1], z_max])
            point_mask = np.logical_and(points >= train_min, points <= train_max).all(axis=1)
            point_indices = np.flatnonzero(point_mask)
            camera_names = sorted(
                name for name, (foot_min, foot_max) in footprints.items()
                if intersects_2d(foot_min, foot_max, train_xy_min, train_xy_max)
            )
            if len(camera_names) < args.min_cameras or point_indices.size == 0:
                print(f"Skipping block {ix:03d}_{iy:03d}: {len(camera_names)} cameras, {point_indices.size} points")
                continue
            block_id = f"{ix:03d}_{iy:03d}"
            block_dir = os.path.join(args.output, f"block_{block_id}")
            os.makedirs(block_dir, exist_ok=True)
            indices_file = os.path.join(block_dir, "point_indices.npy")
            np.save(indices_file, point_indices)
            blocks.append({
                "id": block_id,
                "core_aabb": [[float(core_xy_min[0]), float(core_xy_min[1]), float(z_min)],
                              [float(core_xy_max[0]), float(core_xy_max[1]), float(z_max)]],
                "train_aabb": [train_min.tolist(), train_max.tolist()],
                "core_max_inclusive": [ix == nx - 1, iy == ny - 1, True],
                "train_camera_names": camera_names,
                "test_camera_names": [],
                "point_indices": os.path.relpath(indices_file, args.output),
                "point_count": int(point_indices.size),
                "camera_count": len(camera_names),
            })
            print(f"Block {block_id}: {len(camera_names)} cameras, {point_indices.size} points")

    manifest = {
        "version": 1,
        "source_path": os.path.abspath(args.source_path),
        "images": args.images,
        "grid": [nx, ny],
        "overlap": args.overlap,
        "reference_z": reference_z,
        "scene_aabb": [[float(xy_min[0]), float(xy_min[1]), float(z_min)],
                       [float(xy_max[0]), float(xy_max[1]), float(z_max)]],
        "global_camera_count": len(extrinsics),
        "blocks": blocks,
    }
    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote {len(blocks)} blocks to {manifest_path}")


if __name__ == "__main__":
    main()
