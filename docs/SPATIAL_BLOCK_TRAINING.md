# Spatial block training design

This document describes an incremental implementation of single-GPU spatial
block training for CityGS-X.  The important distinction from the existing
pixel/GPU workload division is that a block owns only a spatial subset of the
point cloud, anchors, densification statistics, and optimizer state.

## Goals and invariants

- Keep the peak memory proportional to the largest block rather than the whole
  city.
- Partition in the shared COLMAP world coordinates used by both `Camera` and
  `GaussianModel`.  The current repository computes a normalization radius but
  does not transform sparse points into a separate per-block coordinate frame.
- Give every block a **core AABB** and a larger **training AABB** (halo).  The
  halo supplies cross-boundary context; only core anchors are emitted when
  blocks are merged.
- Select cameras by view footprint/AABB intersection, not merely by camera
  center.  An aerial camera can observe a block while its center is outside it.
- Preserve global camera UIDs.  Appearance embeddings and checkpoint resume
  must never depend on block-local camera numbering.
- Start with one process invocation per block.  Sequentially swapping optimizer
  states within one process is a later optimization, not the first version.

## Manifest format

`tools/partition_scene.py` creates `blocks/manifest.json`:

```json
{
  "version": 1,
  "grid": [3, 3],
  "overlap": 0.15,
  "scene_aabb": [[-1.0, -0.8, -0.2], [1.1, 1.0, 0.5]],
  "blocks": [
    {
      "id": "000_000",
      "core_aabb": [[-1.0, -0.8, -0.2], [-0.3, -0.2, 0.5]],
      "train_aabb": [[-1.1, -0.9, -0.2], [-0.2, -0.1, 0.5]],
      "train_camera_names": ["DJI_0001", "DJI_0002"],
      "test_camera_names": [],
      "point_indices": "block_000_000/point_indices.npy"
    }
  ]
}
```

Store point indices rather than duplicating PLY files.  This keeps preprocessing
cheap and makes the source COLMAP model the single source of truth.

## Implemented training path

The repository now implements the offline partitioner, manifest loading, camera
and sparse-point filtering, block AABB ownership, AABB-constrained initialization
and densification, block-aware checkpoint validation, and core-only per-block PLY
export.  Each block is trained in an independent process/output folder.

The existing renderer can render one trained block at a time.  Whole-scene
overlap-aware composition and shared-global-MLP refinement remain future phases;
do not concatenate independently trained block models and claim that as a
seam-free whole-scene render.

## Phase 1: offline partitioner

`tools/partition_scene.py` uses the following pipeline:

1. Read COLMAP cameras and sparse points with `scene.colmap_loader`.
2. Keep camera centers and sparse points in their shared COLMAP world frame.
3. Compute robust XY bounds from the 0.5% and 99.5% point quantiles so outliers
   do not create mostly empty blocks.  Keep full-scene Z bounds.
4. Build an `nx * ny` grid and expand each core XY interval by
   `overlap * block_size` to obtain its training AABB.
5. Estimate each camera footprint by intersecting its center ray and four corner
   rays with a reference elevation.  Initially use the median sparse-point Z;
   later replace this with a coarse surface/DEM intersection.
6. Assign a camera when its footprint polygon intersects the training AABB.
   Skip blocks with fewer than configurable `--min_cameras`.  Inspect the
   printed statistics and use a coarser grid if skipping leaves unwanted gaps;
   automatic adjacent-block merging is not implemented yet.
7. Save point indices for points inside the training AABB and write the manifest.

The partitioner must print per-block point count, camera count, and overlap
ratio.  A useful first configuration for CameraA is `3 x 3`, 15% overlap, and at
least 20 cameras per block.  If a block still initializes more than roughly one
million anchors, move to `4 x 4`; do not reduce overlap first.

## Phase 2: arguments and scene filtering

Add these fields to `ModelParams` in `arguments/__init__.py`:

```python
self.block_manifest = ""
self.block_id = ""
self.block_core_only = False
```

Do not put them in `DistributionParams`: blocks describe dataset/model ownership,
whereas the existing distribution options describe runtime rank ownership.

Create `scene/block_partition.py` containing a small, independently testable
`BlockSpec` dataclass and helpers:

```python
@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    core_min: np.ndarray
    core_max: np.ndarray
    train_min: np.ndarray
    train_max: np.ndarray
    train_camera_names: frozenset[str]
    test_camera_names: frozenset[str]
    point_indices_path: str

def load_block_spec(manifest_path: str, block_id: str) -> BlockSpec: ...
def filter_camera_infos(cameras, allowed_names): ...
def filter_basic_point_cloud(point_cloud, indices): ...
```

In `scene/dataset_readers.py`, compute `getNerfppNorm` from the complete camera
set and only then filter cameras and points.  Computing it from filtered cameras
would give blocks inconsistent extents/LOD distance statistics.  `SceneInfo`
therefore carries the global camera count and optional `block_spec`.

In `scene/__init__.py`:

- Load the block specification before decoding images.
- Pass the global camera count to `set_appearance`; retain original camera UIDs.
- Call `save_ply` only with the filtered sparse points.
- Write `block.json` beside `cfg_args` for reproducibility.
- Include block ID in checkpoint validation so a checkpoint cannot accidentally
  be restored into another block.

## Phase 3: block-owned anchors and densification

Initialization is already routed through filtered `points` in `Scene`, so
`set_level` and `create_from_pcd` can remain structurally unchanged.  Add AABB
ownership to `GaussianModel`:

```python
def set_block_bounds(self, core_min, core_max, train_min, train_max): ...

def anchor_inside_train_block(self, xyz):
    return ((xyz >= self.block_train_min) &
            (xyz <= self.block_train_max)).all(dim=-1)

def anchor_inside_core_block(self, xyz):
    return ((xyz >= self.block_core_min) &
            (xyz < self.block_core_max)).all(dim=-1)
```

Use half-open core intervals, except at the global maximum boundary, so every
anchor has exactly one core owner.

In `GaussianModel.anchor_growing`, filter both `candidate_anchor` and
`candidate_anchor_ds` with `anchor_inside_train_block` immediately after their
coordinates are formed and before feature/optimizer tensors are allocated.
Without this guard densification can regrow anchors outside the block and erase
the memory benefit.

Save core/train bounds, block ID, and global camera UID mapping in both the PLY
sidecar and structured training checkpoint.  On final export use the core mask;
on intermediate block resume save the entire training/halo model.

## Phase 4: MLP strategy

Implement the safe version first: each block has its own MLP and output folder.
It requires no cross-process synchronization and establishes whether block
ownership fixes memory usage.

Then add a global-MLP refinement workflow:

1. Train a low-resolution (`--resolution 8`) global model.
2. Initialize every block's MLP from the global checkpoint.
3. Freeze MLPs for the first local stage and optimize anchors/features/offsets.
4. Unfreeze MLPs at a 10x lower learning rate for a short refinement stage.

Never average independent Adam optimizer states across blocks.  If global MLPs
must be jointly trained, use a dedicated epoch loop that visits all blocks,
accumulates only MLP gradients, and takes one global MLP optimizer step.

## Phase 5: training command and orchestration

The first implementation uses one output directory per block:

```bash
python tools/partition_scene.py \
  -s data/CameraA --images train/images \
  --grid 3 3 --overlap 0.15 --output data/CameraA/blocks

python train.py --bsz 1 -s data/CameraA --images train/images \
  --resolution 4 --iterations 30000 \
  --block_manifest data/CameraA/blocks/manifest.json \
  --block_id 000_000 \
  --model_path output/cameraA_blocks/000_000
```

Add `tools/train_blocks.sh` only after one block passes save/resume/render tests.
The script should skip blocks with a valid final checkpoint and run blocks
sequentially on one GPU.  This makes interruption and resume straightforward.

## Phase 6: rendering and merge

Do not concatenate all block anchors onto the GPU; that recreates the original
OOM.  Implement block-aware rendering in two steps:

1. **Offline merge for interchange/mesh:** load one block on CPU, retain only
   core anchors, append them to a CPU PLY, and discard the block.  This is valid
   only when blocks share compatible MLP weights or when exporting a baked
   representation.
2. **Native image rendering:** determine blocks intersecting a camera footprint,
   load them one at a time, render into accumulation buffers, and alpha-compose
   in depth order.  Cache only the last one or two blocks.  For overlapping
   halos, render only core anchors plus a narrow feather band, with deterministic
   spatial weights whose sum is one.

Create `render_blocks.py` rather than adding more branches to `render.py` until
the behavior is stable.  It should accept the manifest and a root model folder,
verify all block configurations/MLP signatures, and write the same output layout
as `render.py`.

## Required tests

Add CPU-only tests for partition geometry and CUDA integration tests separately:

- `tests/test_block_partition.py`: manifest parsing, half-open ownership, halo
  expansion, camera-name filtering, and every point having exactly one core
  owner.
- `tests/test_block_checkpoint.py`: wrong-block restore is rejected; global
  camera UIDs survive save/restore.
- `tests/test_block_densification.py`: all newly grown anchors remain inside the
  training AABB.
- `tests/test_block_render.py`: two overlapping synthetic blocks match a single
  unpartitioned render within tolerance.

For CameraA, record per block: sparse points, initial/final anchors, decoded
camera count, peak allocated/reserved GPU memory, iteration time, and overlap
render error.  Do not accept the implementation solely because it avoids OOM;
compare fixed-camera PSNR/SSIM and inspect seams.

## Recommended implementation order

1. Partitioner + CPU geometry tests.
2. One-block dataset filtering using global normalization and global camera UIDs.
3. AABB-constrained densification and block-aware checkpoint validation.
4. Train/render a single block and compare it with the same crop from the global
   model.
5. Train a `2 x 2` scene and implement core-only CPU merge.
6. Add overlap-aware native rendering.
7. Only then add global shared-MLP refinement and automatic orchestration.

This order avoids a high-risk rewrite of the rasterizer and keeps each stage
measurable and reversible.
