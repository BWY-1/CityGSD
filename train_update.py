import json
import os
from argparse import ArgumentParser

import torch

import train_internal
import utils.general_utils as utils
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
    print_all_args,
)
from utils.general_utils import init_distributed, safe_state


def add_local_update_arguments(parser):
    parser.add_argument("--local_update", action="store_true", default=True)
    parser.add_argument("--update_output_path", required=True)
    parser.add_argument("--update_roi", default="")
    parser.add_argument("--update_roi_file", default="")
    parser.add_argument("--update_roi_margin", type=float, default=0.0)
    parser.add_argument("--update_boundary_lr_scale", type=float, default=0.2)
    parser.add_argument("--min_observation_count", type=int, default=3)
    parser.add_argument("--local_densification", action="store_true")
    parser.add_argument("--save_update_debug", action="store_true")
    shared_group = parser.add_mutually_exclusive_group()
    shared_group.add_argument("--freeze_shared_mlp", dest="freeze_shared_mlp", action="store_true")
    shared_group.add_argument("--train_shared_mlp", dest="freeze_shared_mlp", action="store_false")
    parser.set_defaults(freeze_shared_mlp=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="ROI-guided CityGS-X local scene update")
    ap = AuxiliaryParams(parser)
    # Preserve source-model architecture fields (feature width, offsets, fork,
    # MLP options) from cfg_args unless the user explicitly overrides them.
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dist_p = DistributionParams(parser)
    bench_p = BenchmarkParams(parser)
    debug_p = DebugParams(parser)
    parser.add_argument('--not_use_dpt_loss', action='store_true')
    parser.add_argument('--not_use_single_view_loss', action='store_true')
    parser.add_argument('--not_use_multi_view_loss', action='store_true')
    add_local_update_arguments(parser)

    # model_path points at the old model while get_combined_args reads cfg_args.
    # The output path is installed only after the old architecture/config is loaded.
    args = get_combined_args(parser)
    args.load_model_path = os.path.abspath(args.model_path)
    args.model_path = os.path.abspath(args.update_output_path)
    if args.load_iteration is None:
        args.load_iteration = -1
    if args.min_observation_count < 1:
        parser.error("--min_observation_count must be at least 1")
    if not 0 <= args.update_boundary_lr_scale <= 1:
        parser.error("--update_boundary_lr_scale must be in [0, 1]")
    if args.appearance_dim > 0:
        parser.error("MVP local update requires the source model to use appearance_dim=0")
    if args.not_use_dpt_loss:
        args.dpt_loss_from_iter = args.iterations
    if args.not_use_multi_view_loss:
        args.multi_view_weight_from_iter = args.iterations
    if args.not_use_single_view_loss:
        args.single_view_weight_from_iter = args.iterations

    init_distributed(args)
    if utils.WORLD_SIZE != 1:
        raise NotImplementedError("Distributed local update is not implemented")
    init_args(args)
    args = utils.get_args()
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    log_file = open(os.path.join(args.model_path, "python_ws=1_rk=0.log"), "w")
    utils.set_log_file(log_file)
    print_all_args(args, log_file)
    train_internal.training(
        lp.extract(args), op.extract(args), pp.extract(args), args, log_file
    )
    log_file.close()
    print("Local update complete.")
