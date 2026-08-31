import argparse
import json
import numpy as np
from pyproj import Transformer


def load_scene_cameras(path):
    with open(path, "r") as f:
        cams = json.load(f)

    result = {}

    for cam in cams:
        name = cam["img_name"]

        # cameras.json 通常不带扩展名
        name = name.rsplit(".", 1)[0]

        result[name] = np.array(
            cam["position"],
            dtype=np.float64
        )

    return result


def load_gps_txt(path):
    result = {}

    with open(path, "r", encoding="utf-8-sig") as f:

        header = f.readline().strip().split()

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            name = parts[0].rsplit(".", 1)[0]

            lon = float(parts[1])
            lat = float(parts[2])
            height = float(parts[3])

            result[name] = (lon, lat, height)

    return result


def umeyama_alignment(src, dst):
    """
    Solve:
        dst ~= scale * R @ src + t

    src: Nx3 scene coordinates
    dst: Nx3 metric coordinates
    """

    n = src.shape[0]

    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)

    src_c = src - mean_src
    dst_c = dst - mean_dst

    cov = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)

    # 防止反射
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    var_src = np.mean(
        np.sum(src_c ** 2, axis=1)
    )

    scale = np.trace(
        np.diag(D) @ S
    ) / var_src

    t = mean_dst - scale * (R @ mean_src)

    return scale, R, t


def transform_points(points, scale, R, t):
    return (
        scale * (R @ points.T)
    ).T + t


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cameras",
        required=True
    )

    parser.add_argument(
        "--gps",
        required=True
    )

    parser.add_argument(
        "--scene_area",
        type=float,
        default=15.992009,
        help="Scene convex hull area in scene-units^2"
    )

    args = parser.parse_args()

    scene = load_scene_cameras(args.cameras)
    gps = load_gps_txt(args.gps)

    common = sorted(
        set(scene.keys()) &
        set(gps.keys())
    )

    print("=" * 70)
    print("SCENE -> REAL WORLD SCALE ESTIMATION")
    print("=" * 70)

    print(f"Scene cameras : {len(scene)}")
    print(f"GPS cameras   : {len(gps)}")
    print(f"Matched       : {len(common)}")

    if len(common) < 3:
        raise RuntimeError(
            "Need at least 3 matched cameras."
        )

    # --------------------------------------------------
    # Scene positions
    # --------------------------------------------------

    scene_xyz = np.array(
        [scene[name] for name in common],
        dtype=np.float64
    )

    # --------------------------------------------------
    # GPS -> UTM
    # --------------------------------------------------

    lon = np.array(
        [gps[name][0] for name in common]
    )

    lat = np.array(
        [gps[name][1] for name in common]
    )

    height = np.array(
        [gps[name][2] for name in common]
    )

    mean_lon = lon.mean()
    mean_lat = lat.mean()

    utm_zone = int(
        (mean_lon + 180) / 6
    ) + 1

    if mean_lat >= 0:
        epsg = 32600 + utm_zone
    else:
        epsg = 32700 + utm_zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )

    east, north = transformer.transform(
        lon,
        lat
    )

    real_xyz = np.column_stack(
        [east, north, height]
    )

    # --------------------------------------------------
    # 为了数值稳定，减掉第一个点的大坐标偏移
    # --------------------------------------------------

    real_origin = real_xyz.mean(axis=0)
    real_local = real_xyz - real_origin

    # --------------------------------------------------
    # Sim(3)
    # --------------------------------------------------

    scale, R, t = umeyama_alignment(
        scene_xyz,
        real_local
    )

    pred = transform_points(
        scene_xyz,
        scale,
        R,
        t
    )

    errors = np.linalg.norm(
        pred - real_local,
        axis=1
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    median_error = np.median(errors)
    max_error = np.max(errors)

    # --------------------------------------------------
    # Area conversion
    # --------------------------------------------------

    scene_area = args.scene_area

    area_m2 = (
        scene_area *
        scale ** 2
    )

    area_km2 = area_m2 / 1e6

    print()
    print(f"UTM zone      : {utm_zone}")
    print(f"EPSG          : {epsg}")

    print()
    print("-" * 70)
    print("ESTIMATED SCALE")
    print("-" * 70)

    print(
        f"1 scene unit = "
        f"{scale:.6f} meter"
    )

    print()
    print("-" * 70)
    print("ALIGNMENT ERROR")
    print("-" * 70)

    print(
        f"RMSE          : "
        f"{rmse:.4f} m"
    )

    print(
        f"Median error  : "
        f"{median_error:.4f} m"
    )

    print(
        f"Max error     : "
        f"{max_error:.4f} m"
    )

    print()
    print("-" * 70)
    print("REAL COVERAGE AREA")
    print("-" * 70)

    print(
        f"Scene area    : "
        f"{scene_area:.6f} "
        f"scene-units^2"
    )

    print(
        f"Real area     : "
        f"{area_m2:.2f} m^2"
    )

    print(
        f"Real area     : "
        f"{area_km2:.6f} km^2"
    )

    print()
    print("-" * 70)
    print("TRANSFORMATION")
    print("-" * 70)

    print("Rotation:")
    print(R)

    print()

    print("Translation (local metric coordinates):")
    print(t)

    print()

    print("Metric origin:")
    print(real_origin)

    print("=" * 70)


if __name__ == "__main__":
    main()