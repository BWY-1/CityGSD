# Local Update implementation notes

## Existing data flow

- `Scene` loads cameras from `source_path`, and a trained anchor model plus MLPs
  from `model_path/point_cloud/iteration_*`.
- Anchor rows are `_anchor`, `_offset`, `_anchor_feat`, `_opacity`, `_scaling`,
  and `_rotation`; Adam parameter groups use the corresponding row order.
- `set_anchor_mask` computes the distance/LOD render mask. `prefilter_voxel`
  refines that mask for an individual camera. Local update does not modify either
  render mask.
- The training order is camera sampling, render visibility, forward/render,
  loss, backward, replicated-MLP gradient reduction, densification, and optimizer
  step. Local gradient masking is inserted after any gradient synchronization and
  immediately before the optimizer step.
- Densification collects per-anchor/per-offset statistics in `training_statis`,
  grows rows through `cat_tensors_to_optimizer`, and prunes/reorders rows through
  `_prune_anchor_optimizer`.
- Multi-GPU anchor redistribution performs all-to-all migration of parameters
  and optimizer rows. MVP local update explicitly rejects `WORLD_SIZE > 1` so
  local masks cannot silently become misaligned.

## MVP invariants

- `update_mask = roi_mask & observable_mask`.
- `_anchor_mask` remains a render-only LOD/visibility mask.
- Shared MLPs and appearance embeddings are frozen.
- Static anchor rows are gradient-masked, snapshotted before Adam, and restored
  after Adam. The maximum observed static-row difference is logged.
- Local densification is disabled by default. When enabled, only update anchors
  contribute statistics; new rows inherit local state and pruning is restricted
  to update rows.
