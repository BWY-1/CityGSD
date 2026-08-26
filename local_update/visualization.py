import json
import os

import numpy as np
from plyfile import PlyData, PlyElement


def _write_mask_ply(path, positions, mask, color):
    points = positions[mask].detach().cpu().numpy().astype(np.float32)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertices = np.empty(points.shape[0], dtype=dtype)
    if points.shape[0]:
        vertices["x"], vertices["y"], vertices["z"] = points.T
        vertices["red"], vertices["green"], vertices["blue"] = color
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def save_local_update_debug(output_dir, gaussians, context):
    os.makedirs(output_dir, exist_ok=True)
    positions = gaussians.get_anchor_centers()
    _write_mask_ply(
        os.path.join(output_dir, "all_anchors.ply"),
        positions,
        context.update_mask.new_ones(context.update_mask.shape),
        (180, 180, 180),
    )
    _write_mask_ply(os.path.join(output_dir, "roi_mask.ply"), positions, context.roi_mask, (255, 180, 0))
    _write_mask_ply(os.path.join(output_dir, "observable_mask.ply"), positions, context.observable_mask, (0, 180, 255))
    _write_mask_ply(os.path.join(output_dir, "update_mask.ply"), positions, context.update_mask, (255, 0, 0))
    np.savez_compressed(
        os.path.join(output_dir, "anchor_states.npz"),
        visible_count=context.visible_count.cpu().numpy(),
        core_mask=context.core_mask.cpu().numpy(),
        boundary_mask=context.boundary_mask.cpu().numpy(),
        roi_mask=context.roi_mask.cpu().numpy(),
        observable_mask=context.observable_mask.cpu().numpy(),
        update_mask=context.update_mask.cpu().numpy(),
    )
    metadata = {
        "num_total_anchors": int(context.update_mask.numel()),
        "num_roi_anchors": int(context.roi_mask.sum().item()),
        "num_observable_anchors": int(context.observable_mask.sum().item()),
        "num_update_anchors": int(context.update_mask.sum().item()),
        "update_ratio": context.update_ratio,
        "min_observation_count": context.min_observation_count,
        "max_static_difference_before_restore": context.max_static_difference,
        "max_static_difference_after_restore": 0.0,
    }
    with open(os.path.join(output_dir, "update_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata
