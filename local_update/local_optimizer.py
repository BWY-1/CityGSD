import torch


ANCHOR_GROUPS = ("anchor", "offset", "anchor_feat", "opacity", "scaling", "rotation")
SHARED_GROUPS = (
    "mlp_opacity",
    "mlp_cov",
    "mlp_color",
    "mlp_featurebank",
    "embedding_appearance",
)


def freeze_shared_modules(gaussians):
    modules = [gaussians.mlp_opacity, gaussians.mlp_cov, gaussians.mlp_color]
    if gaussians.use_feat_bank:
        modules.append(gaussians.mlp_feature_bank)
    if gaussians.appearance_dim > 0 and gaussians.embedding_appearance is not None:
        modules.append(gaussians.embedding_appearance)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    for group in gaussians.optimizer.param_groups:
        if group["name"] in SHARED_GROUPS:
            group["lr"] = 0.0


def capture_static_rows(param_groups, update_mask):
    static_mask = ~update_mask
    snapshot = {"mask": static_mask}
    for group in param_groups:
        if group["name"] in ANCHOR_GROUPS:
            snapshot[group["name"]] = group["params"][0].detach()[static_mask].clone()
    return snapshot


@torch.no_grad()
def restore_static_rows(param_groups, snapshot):
    static_mask = snapshot["mask"]
    max_difference = 0.0
    for group in param_groups:
        name = group["name"]
        if name not in snapshot:
            continue
        parameter = group["params"][0]
        if snapshot[name].numel():
            difference = (parameter[static_mask] - snapshot[name]).abs().max().item()
            max_difference = max(max_difference, difference)
            parameter[static_mask] = snapshot[name]
    return max_difference
