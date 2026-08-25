from dataclasses import dataclass

import torch


@dataclass
class LocalUpdateContext:
    enabled: bool
    roi_mask: torch.Tensor
    observable_mask: torch.Tensor
    core_mask: torch.Tensor
    boundary_mask: torch.Tensor
    update_mask: torch.Tensor
    visible_count: torch.Tensor
    min_observation_count: int
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    roi_margin: float
    max_static_difference: float = 0.0

    @property
    def update_ratio(self):
        if self.update_mask.numel() == 0:
            return 0.0
        return float(self.update_mask.float().mean().item())


def build_local_update_context(args, gaussians, cameras, pipe_args, background, iteration):
    from .observable_region import compute_observable_region
    from .roi import ROISelector, parse_bbox

    bbox_min, bbox_max = parse_bbox(args.update_roi, args.update_roi_file)
    selector = ROISelector(bbox_min, bbox_max, args.update_roi_margin)
    centers = gaussians.get_anchor_centers()
    core_mask, boundary_mask, roi_mask = selector.build_masks(centers)
    visible_count, observable_mask = compute_observable_region(
        cameras,
        gaussians,
        pipe_args,
        background,
        args.min_observation_count,
        iteration,
    )
    update_mask = roi_mask & observable_mask
    if not update_mask.any():
        raise ValueError(
            "The ROI contains no anchors observed by the new cameras at the "
            "requested min_observation_count"
        )
    context = LocalUpdateContext(
        enabled=True,
        roi_mask=roi_mask,
        observable_mask=observable_mask,
        core_mask=core_mask,
        boundary_mask=boundary_mask,
        update_mask=update_mask,
        visible_count=visible_count,
        min_observation_count=args.min_observation_count,
        bbox_min=bbox_min.to(centers.device),
        bbox_max=bbox_max.to(centers.device),
        roi_margin=args.update_roi_margin,
    )
    gaussians.initialize_local_update_state(context, args.update_boundary_lr_scale)
    return context


def synchronize_context_from_model(context, gaussians):
    """Refresh context tensors after local densification appends or prunes rows."""
    context.roi_mask = gaussians._roi_mask
    context.observable_mask = gaussians._observable_mask
    context.core_mask = gaussians._local_core_mask
    context.boundary_mask = gaussians._local_boundary_mask
    context.update_mask = gaussians._local_update_mask
    context.visible_count = gaussians._local_visible_count
