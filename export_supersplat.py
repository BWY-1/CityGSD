"""Bake a view-dependent CityGSD checkpoint into a standard 3DGS PLY."""

import os
from argparse import ArgumentParser

import torch

from arguments import (
    AuxiliaryParams,
    BenchmarkParams,
    DebugParams,
    DistributionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
    init_args,
)
from gaussian_renderer import GaussianModel, generate_neural_gaussians
from scene import Scene
from utils.general_utils import (
    init_distributed,
    inverse_sigmoid,
    safe_state,
    set_args,
    set_log_file,
)
from utils.sh_utils import RGB2SH
import utils.general_utils as utils


def _resolve_source_path(source_path):
    """Accept either the scene root or its ``train`` directory.

    ``Scene`` expects the directory containing ``sparse``.  It is easy to pass
    ``<scene>/train`` because the RGB images live below that directory, so give
    that common mistake a useful automatic correction and otherwise fail with
    an actionable message before the more generic Scene error.
    """
    source_path = os.path.abspath(source_path)
    if os.path.isdir(os.path.join(source_path, "sparse")):
        return source_path

    parent_path = os.path.dirname(source_path)
    if os.path.basename(source_path).lower() == "train" and os.path.isdir(
        os.path.join(parent_path, "sparse")
    ):
        print(
            f"Source path {source_path!r} is the train directory; "
            f"using scene root {parent_path!r} instead."
        )
        return parent_path

    if "MatrixCity" in source_path:
        return source_path

    raise ValueError(
        f"Invalid source path {source_path!r}: expected a scene root containing "
        "a 'sparse' directory. Pass -s <scene_root>, not -s <scene_root>/train; "
        "select the image subdirectory separately with --images train/rgbs."
    )


def _resolve_images_path(source_path, images_path):
    """Validate the image directory and correct common ``rgbs``/``images`` mixups."""
    requested_path = (
        images_path if os.path.isabs(images_path) else os.path.join(source_path, images_path)
    )
    if os.path.isdir(requested_path):
        return images_path

    common_paths = ("train/images", "train/rgbs", "images", "rgbs")
    matches = [
        candidate
        for candidate in common_paths
        if os.path.isdir(os.path.join(source_path, candidate))
    ]
    if len(matches) == 1:
        print(
            f"Image directory {requested_path!r} does not exist; "
            f"using {matches[0]!r} instead."
        )
        return matches[0]

    available = ", ".join(repr(path) for path in matches) or "none"
    raise ValueError(
        f"Image directory {requested_path!r} does not exist. Set --images to a "
        f"directory relative to the scene root {source_path!r}. Detected common "
        f"image directories: {available}."
    )


def _select_camera(cameras, camera_index, camera_name):
    if not cameras:
        raise ValueError("The selected camera split is empty")
    if camera_name:
        matches = [camera for camera in cameras if camera.image_name == camera_name]
        if not matches:
            available = ", ".join(camera.image_name for camera in cameras[:5])
            raise ValueError(
                f"Camera {camera_name!r} was not found; the first cameras are: {available}"
            )
        return matches[0]
    index = len(cameras) // 2 if camera_index is None else camera_index
    if not -len(cameras) <= index < len(cameras):
        raise IndexError(f"camera-index {index} is outside [0, {len(cameras) - 1}]")
    return cameras[index]


def _property_names(sh_degree):
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(3 * ((sh_degree + 1) ** 2 - 1))]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]
    return names


def _standard_3dgs_records(xyz, color, opacity, scaling, rotation, sh_degree):
    """Convert activated renderer values to the values stored by standard 3DGS."""
    eps = 1e-6
    xyz = xyz.float()
    color = color.float().clamp(0.0, 1.0)
    opacity = opacity.float().clamp(eps, 1.0 - eps)
    scaling = scaling.float().clamp_min(eps)
    rotation = torch.nn.functional.normalize(rotation.float(), dim=1)

    columns = [xyz, torch.zeros_like(xyz), RGB2SH(color)]
    rest_count = 3 * ((sh_degree + 1) ** 2 - 1)
    if rest_count:
        columns.append(torch.zeros((xyz.shape[0], rest_count), device=xyz.device))
    columns += [inverse_sigmoid(opacity), torch.log(scaling), rotation]
    return torch.cat(columns, dim=1).detach().cpu().numpy().astype("<f4", copy=False)


_PLY_COUNT_WIDTH = 20


def _write_ply_header(output, property_names):
    """Write a patchable PLY header and return the vertex-count byte offset."""
    count_placeholder = "0" * _PLY_COUNT_WIDTH
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count_placeholder}"]
    header += [f"property float {name}" for name in property_names]
    header += ["end_header", ""]
    header_bytes = "\n".join(header).encode("ascii")
    count_offset = header_bytes.index(count_placeholder.encode("ascii"))
    output.write(header_bytes)
    return count_offset


def _patch_ply_count(output, count_offset, count):
    count_bytes = f"{count:0{_PLY_COUNT_WIDTH}d}".encode("ascii")
    if len(count_bytes) != _PLY_COUNT_WIDTH:
        raise ValueError(f"PLY vertex count {count} exceeds {_PLY_COUNT_WIDTH} digits")
    output.seek(count_offset)
    output.write(count_bytes)
    output.flush()


def _remove_incomplete_output(output_path):
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass


def _raise_disk_error(output_path, error):
    if getattr(error, "errno", None) == 28:
        _remove_incomplete_output(output_path)
        raise OSError(
            28,
            "Not enough disk space for the final PLY. The incomplete output was "
            "removed; increase --min-opacity or choose an output on a larger disk.",
            output_path,
        ) from error
    raise error


def _open_ply_output(output_path, property_names):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output = open(output_path, "wb+")
    try:
        return output, _write_ply_header(output, property_names)
    except Exception:
        output.close()
        raise


@torch.no_grad()
def bake_gaussians(gaussians, camera, output_path, chunk_size, min_opacity, sh_degree):
    property_names = _property_names(sh_degree)
    anchor_count = gaussians.get_anchor.shape[0]
    written = 0

    output = None
    try:
        output, count_offset = _open_ply_output(output_path, property_names)
        try:
            for start in range(0, anchor_count, chunk_size):
                end = min(start + chunk_size, anchor_count)
                visible_mask = torch.zeros(anchor_count, dtype=torch.bool, device="cuda")
                visible_mask[start:end] = True
                xyz, color, opacity, scaling, rotation = generate_neural_gaussians(
                    camera, gaussians, visible_mask=visible_mask, is_training=False
                )
                keep = opacity[:, 0] >= min_opacity
                if keep.any():
                    records = _standard_3dgs_records(
                        xyz[keep], color[keep], opacity[keep], scaling[keep],
                        rotation[keep], sh_degree
                    )
                    output.write(records.tobytes(order="C"))
                    written += records.shape[0]
                print(f"Baked anchors {end:,}/{anchor_count:,}; kept {written:,} Gaussians")
            _patch_ply_count(output, count_offset, written)
        except OSError as error:
            output.close()
            output = None
            _raise_disk_error(output_path, error)
    except Exception:
        _remove_incomplete_output(output_path)
        raise
    finally:
        if output is not None:
            output.close()
    return written


def export(args, dataset):
    args.source_path = _resolve_source_path(args.source_path)
    args.images = _resolve_images_path(args.source_path, args.images)
    dataset.source_path = args.source_path
    dataset.images = args.images

    # get_combined_args deliberately omits command-line values whose default is
    # None.  Older cfg_args files do not contain these exporter-only options,
    # so optional arguments must be read with getattr rather than assumed to be
    # present on the merged Namespace.
    camera_index = getattr(args, "camera_index", None)
    camera_name = getattr(args, "camera_name", None)
    camera_split = getattr(args, "camera_split", "train")

    # Scene decodes every requested image even though baking only needs one
    # camera.  For a non-negative numeric index, load only the prefix necessary
    # to reach it. Name and negative-index selection still require the full set.
    if camera_name is None and camera_index is not None and camera_index >= 0:
        if camera_split == "test":
            args.num_test_cameras = camera_index + 1
        else:
            args.num_train_cameras = camera_index + 1

    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.add_opacity_dist, dataset.add_cov_dist,
        dataset.add_color_dist, dataset.add_level, dataset.visible_threshold,
        dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend,
    )
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTestCameras() if camera_split == "test" else scene.getTrainCameras()
    camera = _select_camera(cameras, camera_index, camera_name)
    gaussians.eval()
    count = bake_gaussians(
        gaussians, camera, args.output, args.chunk_size, args.min_opacity, args.sh_degree
    )
    print(f"Wrote {count:,} Gaussians baked from camera {camera.image_name!r} to {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Bake CityGSD neural Gaussians for SuperSplat")
    AuxiliaryParams(parser)
    lp = ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    DistributionParams(parser)
    BenchmarkParams(parser)
    DebugParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output", required=True, help="Output standard 3DGS PLY")
    parser.add_argument("--camera-split", choices=("train", "test"), default="train")
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument("--camera-index", type=int, default=None)
    camera_group.add_argument("--camera-name", default=None)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--min-opacity", type=float, default=0.01)
    parser.add_argument("--sh-degree", type=int, choices=range(4), default=0)
    parser.add_argument("--distributed_load", action="store_true")
    parser.add_argument("--not_use_dpt_loss", action="store_false")
    parser.add_argument("--not_use_single_view_loss", action="store_false")
    parser.add_argument("--not_use_multi_view_loss", action="store_false")
    args = get_combined_args(parser)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if not 0.0 <= args.min_opacity < 1.0:
        parser.error("--min-opacity must be in [0, 1)")

    init_distributed(args)
    if utils.WORLD_SIZE != 1:
        raise RuntimeError("export_supersplat.py currently supports exactly one GPU")
    args.distributed_dataset_storage = False
    args.num_test_cameras = -1
    args.num_train_cameras = -1
    log_file = open(os.path.join(args.model_path, "export_supersplat.log"), "w")
    set_log_file(log_file)
    init_args(args)
    set_args(args)
    safe_state(args.quiet)
    export(args, lp.extract(args))