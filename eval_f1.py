import argparse
import json

import numpy as np


def summarize_distances(pred_to_gt, gt_to_pred, threshold):
    pred_to_gt = np.asarray(pred_to_gt, dtype=np.float64)
    gt_to_pred = np.asarray(gt_to_pred, dtype=np.float64)
    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        return {
            "threshold": float(threshold),
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
        }

    precision = float(np.mean(pred_to_gt < threshold))
    recall = float(np.mean(gt_to_pred < threshold))
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "accuracy_mean": float(pred_to_gt.mean()),
        "accuracy_median": float(np.median(pred_to_gt)),
        "accuracy_p95": float(np.quantile(pred_to_gt, 0.95)),
        "completeness_mean": float(gt_to_pred.mean()),
        "completeness_median": float(np.median(gt_to_pred)),
        "completeness_p95": float(np.quantile(gt_to_pred, 0.95)),
        "chamfer_l1": float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())),
    }


def load_geometry(path, sample_points, voxel_size):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > 0:
        mesh.remove_unreferenced_vertices()
        point_cloud = mesh.sample_points_uniformly(number_of_points=sample_points)
        source_kind = "triangle_mesh"
    else:
        point_cloud = o3d.io.read_point_cloud(path)
        source_kind = "point_cloud"
    if voxel_size > 0:
        point_cloud = point_cloud.voxel_down_sample(voxel_size)
    if len(point_cloud.points) == 0:
        raise ValueError(f"No points could be loaded from {path}")
    return point_cloud, source_kind


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstructed geometry against aligned ground truth")
    parser.add_argument("--ply_path_pred", required=True)
    parser.add_argument("--ply_path_gt", required=True)
    parser.add_argument(
        "--dtau",
        "--distance_threshold",
        dest="distance_threshold",
        type=float,
        required=True,
        help="Distance threshold in the coordinate unit of the input geometries",
    )
    parser.add_argument("--sample_points", type=int, default=2_000_000)
    parser.add_argument("--voxel_size", type=float, default=0.0)
    parser.add_argument(
        "--legacy_half_threshold",
        action="store_true",
        help="Use dtau/2 to reproduce the previous evaluator behavior",
    )
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()
    if args.distance_threshold <= 0 or args.sample_points <= 0 or args.voxel_size < 0:
        parser.error("threshold and sample_points must be positive; voxel_size cannot be negative")

    pred, pred_kind = load_geometry(args.ply_path_pred, args.sample_points, args.voxel_size)
    gt, gt_kind = load_geometry(args.ply_path_gt, args.sample_points, args.voxel_size)
    pred_to_gt = pred.compute_point_cloud_distance(gt)
    gt_to_pred = gt.compute_point_cloud_distance(pred)
    threshold = args.distance_threshold / 2.0 if args.legacy_half_threshold else args.distance_threshold
    metrics = summarize_distances(pred_to_gt, gt_to_pred, threshold)
    metrics.update({
        "prediction": args.ply_path_pred,
        "ground_truth": args.ply_path_gt,
        "prediction_type": pred_kind,
        "ground_truth_type": gt_kind,
        "prediction_points": len(pred.points),
        "ground_truth_points": len(gt.points),
    })

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
