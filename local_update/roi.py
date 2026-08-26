import json

import torch


def parse_bbox(value="", file_path=""):
    if bool(value) == bool(file_path):
        raise ValueError("Specify exactly one of --update_roi or --update_roi_file")
    if file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        try:
            data = json.loads(raw)
            values = data.get("bbox", data) if isinstance(data, dict) else data
        except json.JSONDecodeError:
            values = [float(part) for part in raw.replace(",", " ").split()]
    else:
        values = [float(part.strip()) for part in value.split(",")]
    if len(values) != 6:
        raise ValueError("ROI bbox must contain xmin,ymin,zmin,xmax,ymax,zmax")
    bbox_min = torch.tensor(values[:3], dtype=torch.float32)
    bbox_max = torch.tensor(values[3:], dtype=torch.float32)
    if not torch.all(bbox_max > bbox_min):
        raise ValueError("Every ROI maximum must be greater than its minimum")
    return bbox_min, bbox_max


class ROISelector:
    def __init__(self, bbox_min, bbox_max, margin=0.0):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.margin = float(margin)
        if self.margin < 0:
            raise ValueError("ROI margin cannot be negative")

    @staticmethod
    def _inside(anchor_positions, lower, upper):
        return ((anchor_positions >= lower) & (anchor_positions <= upper)).all(dim=-1)

    def build_masks(self, anchor_positions):
        lower = self.bbox_min.to(anchor_positions.device)
        upper = self.bbox_max.to(anchor_positions.device)
        core_mask = self._inside(anchor_positions, lower, upper)
        expanded_mask = self._inside(
            anchor_positions, lower - self.margin, upper + self.margin
        )
        boundary_mask = expanded_mask & ~core_mask
        return core_mask, boundary_mask, expanded_mask
