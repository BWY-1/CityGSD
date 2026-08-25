import torch

from gaussian_renderer import prefilter_voxel


@torch.no_grad()
def compute_observable_region(
    cameras,
    gaussians,
    pipe_args,
    background,
    min_observation_count=3,
    iteration=0,
):
    visible_count = torch.zeros(
        gaussians.get_anchor.shape[0], dtype=torch.int32, device="cuda"
    )
    for camera in cameras:
        gaussians.set_anchor_mask(camera.camera_center, iteration, 1)
        visible = prefilter_voxel(camera, gaussians, pipe_args, background)
        if visible.shape != visible_count.shape:
            raise ValueError(
                f"Visibility mask shape {tuple(visible.shape)} does not match "
                f"anchor shape {tuple(visible_count.shape)}"
            )
        visible_count += visible.to(torch.int32)
    observable_mask = visible_count >= min_observation_count
    return visible_count, observable_mask
