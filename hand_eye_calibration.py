"""
hand_eye_calibration.py

Entry point for hand-eye (eye-in-hand) calibration.
All functionality is imported from calib_module.py in the same directory,
so no crp_calibration / crp_core package installation is required.

File naming convention: {index}_{tx}_{ty}_{tz}_{rx}_{ry}_{rz}.ext

Uses the same data structure and CLI as fixed_eye_calibration.py.
The reported transforms differ: flange2cam / base2world (vs base2cam / world2flange).

Supported input types:
  rgb       : .png / .jpg / .jpeg / .bmp  (PnP-based pose)
  ply       : .ply                         (3D SVD-based pose)
  rgb_depth : RGB image + .ply depth       (2D corners + 3D lookup)

Example:
  python hand_eye_calibration.py \\
      --data_dir ./images \\
      --rows 8 --cols 7 --size 30 \\
      --input_type ply

  # RGB mode (uses built-in intrinsics; *_intrinsic.json is applied automatically if present)
  python hand_eye_calibration.py \\
      --data_dir ./images --input_type rgb \\
      --dist_coeffs 0 0 0 0 0
"""

# ─────────────────────────────────────────────────────────────────────────────
#  User defaults — set paths, board, and pose conventions here.
#  All values can be overridden via CLI arguments.
# ─────────────────────────────────────────────────────────────────────────────
USER_DEFAULT_DATA_DIR = "data/260413"
USER_DEFAULT_OUTPUT_DIR = "./calibration_output_handeye"
USER_DEFAULT_INPUT_TYPE = "rgb"          # "rgb" | "ply" | "rgb_depth"
USER_DEFAULT_ROWS = 8
USER_DEFAULT_COLS = 7
USER_DEFAULT_SIZE = 30.0                 # checkerboard square size (mm)
USER_DEFAULT_ROT_TYPE = "euler"          # euler | rotvec | quat
USER_DEFAULT_ROT_UNIT = "deg"            # deg | rad
USER_DEFAULT_POS_UNIT = "mm"             # mm | m
USER_DEFAULT_EULER_CONV = "xyz"
USER_DEFAULT_DEBUG = False
USER_DEFAULT_REFERENCE = None            # path to a reference .npz or parent directory
USER_DEFAULT_INTRINSIC = (
    616.178466796875,
    616.2945556640625,
    326.4101867675781,
    239.1746368408203,
)
USER_DEFAULT_DIST_COEFFS = None          # None → 5 zeros

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# calib_module lives in the same directory; insert it into sys.path if needed.
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from calib_module import (
    Logger,
    HandEyeCalibration,
    CameraEstimator,
    Frames,
    Frame,
    CalibrationError,
    inv_se3,
    get_se3,
)


# ──────────────────────────────────────────────
#  Image I/O (Unicode path support)
# ──────────────────────────────────────────────

def imread_unicode(path: str) -> np.ndarray:
    """Read an image from a Unicode path."""
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Write an image to a Unicode path."""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


# ──────────────────────────────────────────────
#  Comparison helpers
# ──────────────────────────────────────────────

def pose_translation_distance(T1: np.ndarray, T2: np.ndarray) -> float:
    return float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]))


def pose_angular_distance_deg(T1: np.ndarray, T2: np.ndarray) -> float:
    R_rel = T1[:3, :3] @ T2[:3, :3].T
    trace = np.trace(R_rel)
    angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle_rad))


# ──────────────────────────────────────────────
#  Filename parser: {index}_{tx}_{ty}_{tz}_{rx}_{ry}_{rz}.ext
# ──────────────────────────────────────────────

def parse_filename_pose(filename, rot_type="euler", rot_unit="deg",
                        pos_unit="mm", euler_conv="xyz"):
    """Parse index and base2flange SE3 (4×4) from a filename."""
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 7:
        raise ValueError(f"Need ≥7 fields in '{stem}', got {len(parts)}")

    idx = int(parts[0])
    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
    r1, r2, r3 = float(parts[4]), float(parts[5]), float(parts[6])

    t = np.array([tx, ty, tz], dtype=np.float64)
    if pos_unit == "m":
        t *= 1000.0

    if rot_type == "euler":
        R_mat = Rot.from_euler(euler_conv, [r1, r2, r3],
                               degrees=(rot_unit == "deg")).as_matrix()
    elif rot_type == "rotvec":
        rv = np.array([r1, r2, r3])
        if rot_unit == "deg":
            rv = np.deg2rad(rv)
        R_mat = Rot.from_rotvec(rv).as_matrix()
    elif rot_type == "quat" and len(parts) >= 8:
        R_mat = Rot.from_quat([r1, r2, r3, float(parts[7])]).as_matrix()
    else:
        raise ValueError(f"Unsupported rot_type: {rot_type}")

    return idx, get_se3(R_mat, t)


def load_data_dir_intrinsics_json(path: str):
    """Read K and distortion coefficients from data_dir/intrinsics.json. Returns (None, None) on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None

    K = None
    if "K" in data:
        K = np.array(data["K"], dtype=np.float64).reshape(3, 3)
    ins = data.get("intrinsics") or {}
    if K is None and all(k in ins for k in ("fx", "fy", "ppx", "ppy")):
        K = np.array(
            [
                [float(ins["fx"]), 0.0, float(ins["ppx"])],
                [0.0, float(ins["fy"]), float(ins["ppy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if K is None:
        return None, None

    coeffs = ins.get("coeffs")
    dist = (
        np.array(coeffs, dtype=np.float64).ravel()
        if coeffs and len(coeffs) > 0
        else np.zeros(5, dtype=np.float64)
    )
    return K, dist


# ──────────────────────────────────────────────
#  Main run function
# ──────────────────────────────────────────────

def run(args):
    logger = Logger(
        name="HandEyeCalibration",
        project_name="CRP_Calibration",
        log_path=args.output_dir,
    )
    logger.debug_mode = args.debug

    # Resolve relative paths relative to the script's directory.
    _script_root = Path(__file__).resolve().parent
    for attr in ("data_dir", "output_dir"):
        p = Path(getattr(args, attr))
        if not p.is_absolute():
            p = _script_root / p
        setattr(args, attr, str(p))

    logger.log_info(f"Board: {args.rows}×{args.cols}, sq={args.size}mm")
    logger.log_info(f"Pose: {args.rot_type}({args.rot_unit}), {args.pos_unit}, euler={args.euler_conv}")
    logger.log_info(f"Input type: {args.input_type}")

    # Collect input files.
    files = []
    if args.input_type == "ply":
        files.extend(glob.glob(os.path.join(args.data_dir, "*.ply")))
    else:
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            files.extend(glob.glob(os.path.join(args.data_dir, ext)))
    files = sorted(files)

    if not files:
        ftype = "PLY files" if args.input_type == "ply" else "images"
        logger.log_error(f"No {ftype} in {args.data_dir}")
        return None
    logger.log_info(f"Found {len(files)} file(s)\n")

    # Build parameter dict.
    param = {"rows": args.rows, "cols": args.cols, "size": args.size}

    bundle_json = os.path.join(args.data_dir, "intrinsics.json")
    if os.path.isfile(bundle_json):
        K_b, dist_b = load_data_dir_intrinsics_json(bundle_json)
        if K_b is not None:
            param["camera_matrix"] = K_b
            param["dist_coeffs"] = dist_b
            logger.log_info(
                f"data_dir intrinsics: {bundle_json} "
                f"(fx={K_b[0,0]:.4f}, fy={K_b[1,1]:.4f}, "
                f"cx={K_b[0,2]:.4f}, cy={K_b[1,2]:.4f}; "
                f"per-file *_intrinsic.json overrides when present)"
            )

    if "camera_matrix" not in param and args.intrinsic is not None:
        fx, fy, cx, cy = args.intrinsic
        param["camera_matrix"] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        param["dist_coeffs"] = (
            np.array(args.dist_coeffs, dtype=np.float64)
            if args.dist_coeffs else np.zeros(5, dtype=np.float64)
        )
        logger.log_info(
            f"CLI intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy} "
            f"(per-file *_intrinsic.json overrides when present)"
        )

    if args.input_type == "ply":
        param["ply_render_use_real_camera"] = True

    estimator = CameraEstimator(logger, param)
    calibration = HandEyeCalibration(logger, param)
    frames = Frames(logger)

    os.makedirs(args.output_dir, exist_ok=True)
    robot_pose_type = SimpleNamespace(
        rot_type=args.rot_type,
        rot_unit=args.rot_unit,
        scale_unit=args.pos_unit,
    )
    info = []  # list of (idx, fname) for logging

    logger.log_info("=======================================")
    logger.log_info("Phase 1: Detection + PnP (via estimator.get_pose)")
    logger.log_info("=======================================")

    for fpath in files:
        fname = os.path.basename(fpath)

        try:
            idx, _ = parse_filename_pose(
                fname,
                rot_type=args.rot_type,
                rot_unit=args.rot_unit,
                pos_unit=args.pos_unit,
                euler_conv=args.euler_conv,
            )
        except Exception as e:
            logger.log_warn(f"  Skip {fname}: {e}")
            continue

        try:
            file_list = [fpath]
            if args.input_type == "rgb_depth":
                ply_path = os.path.splitext(fpath)[0] + ".ply"
                if os.path.exists(ply_path):
                    file_list.append(ply_path)

            # Apply per-frame *_intrinsic.json if present.
            if args.input_type in ("rgb", "rgb_depth"):
                intr_path = os.path.splitext(fpath)[0] + "_intrinsic.json"
                if os.path.exists(intr_path):
                    with open(intr_path, "r") as _f:
                        _intr = json.load(_f)
                    _sensor = _intr
                    if "sensores" in _intr and "image" in _intr["sensores"]:
                        _sensor = _intr["sensores"]["image"]
                    if "intrinsic_matrix" in _sensor:
                        estimator.real_camera_matrix = np.array(
                            _sensor["intrinsic_matrix"], dtype=np.float64
                        ).reshape(3, 3)
                    if "distortion_coefficients" in _sensor:
                        estimator.real_dist_coeffs = np.array(
                            _sensor["distortion_coefficients"], dtype=np.float64
                        )
                    if "resolution" in _sensor:
                        estimator.width = _sensor["resolution"]["width"]
                        estimator.height = _sensor["resolution"]["height"]

            robot, camera, succ_id, vis_bgr, pcd, marker_point = estimator.get_pose(
                file_list, idx, robot_pose_type, input_type=args.input_type
            )

        except CalibrationError as e:
            logger.log_warn(f"  Skip {fname}: {e}")
            continue

        # In PLY / rgb_depth mode, invert the camera pose before storing.
        if args.input_type in ("ply", "rgb_depth"):
            camera = inv_se3(camera)

        frame = Frame(idx, robot=robot, camera=camera)
        frames.add_frame(frame)
        logger.log_info(f"  [{idx:3d}] t={camera[:3,3].round(1)}  {fname}")
        info.append((idx, fname))

        if vis_bgr is not None:
            imwrite_unicode(
                os.path.join(args.output_dir, f"{idx:03d}_detected.png"), vis_bgr
            )

    logger.log_info(f"\n{len(frames)}/{len(files)} valid\n")

    if len(frames) < 3:
        logger.log_error(f"Need ≥3 valid frames, got {len(frames)}")
        return None

    logger.log_info("=======================================")
    logger.log_info("Phase 2: Calibration (HandEyeCalibration)")
    logger.log_info("=======================================")

    result = calibration.execute(frames)

    logger.log_info("\n=======================================")
    logger.log_info("RESULTS")
    logger.log_info("=======================================")

    _fmt = lambda arr: np.array2string(
        arr, suppress_small=True, precision=4,
        formatter={"float_kind": lambda x: f"{x:10.4f}"},
    )

    def _fmt_yml_matrix(key: str, arr: np.ndarray) -> str:
        rows = arr.tolist()
        lines = [f"    {key}: [" + repr(rows[0]) + ","]
        for r in rows[1:-1]:
            lines.append("            " + repr(r) + ",")
        lines.append("            " + repr(rows[-1]) + "]")
        return "\n".join(lines)

    flange2cam = result["flange2cam"]
    euler_f2c = Rot.from_matrix(flange2cam[:3, :3]).as_euler("xyz", degrees=True)

    # Save result in picking_zone.yml format.
    yml_lines = [
        "config:",
        "    marker_mode: Tool",
        "    estimator_mode: Checkerboard",
        "    calibration_mode: HandEye",
        "    marker_id: []",
        f"    size: {args.size}",
        f"    rows: {args.rows}",
        f"    cols: {args.cols}",
        "    error: 0.0",
    ]
    for key in ("X", "Y", "flange2cam", "cam2flange", "base2world", "world2base"):
        if key in result:
            yml_lines.append(_fmt_yml_matrix(key, result[key]))
    yml_lines.append(f"    score: {result['score']}")

    fr = result.get("frame_results", [])
    fr_list = fr.tolist() if hasattr(fr, "tolist") else list(fr)
    yml_lines.append("    frames_error: " + repr(fr_list))

    valid_ids_set = set(result.get("valid_ids", []))
    rejected_ids = sorted([idx for idx, _ in info if idx not in valid_ids_set])
    yml_lines.append("    rejected_frame_ids: " + repr(rejected_ids))

    yml_path = os.path.join(args.output_dir, "calibration_result.yml")
    with open(yml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yml_lines) + "\n")
    logger.log_info(f"Saved calibration (picking_zone format) → {yml_path}")

    logger.log_info(f"\nflange2cam (X):\n{_fmt(flange2cam)}")
    logger.log_info(f"  t(mm): [{flange2cam[0,3]:.0f}, {flange2cam[1,3]:.0f}, {flange2cam[2,3]:.0f}]")
    logger.log_info(f"  R(deg): [{euler_f2c[0]:.2f}, {euler_f2c[1]:.2f}, {euler_f2c[2]:.2f}]")

    logger.log_info(f"\ncam2flange:\n{_fmt(result['cam2flange'])}")

    base2world = result["base2world"]
    euler_b2w = Rot.from_matrix(base2world[:3, :3]).as_euler("xyz", degrees=True)
    logger.log_info(f"\nbase2world (Y):\n{_fmt(base2world)}")
    logger.log_info(f"  t(mm): [{base2world[0,3]:.0f}, {base2world[1,3]:.0f}, {base2world[2,3]:.0f}]")
    logger.log_info(f"  R(deg): [{euler_b2w[0]:.2f}, {euler_b2w[1]:.2f}, {euler_b2w[2]:.2f}]")

    logger.log_info(f"\nworld2base:\n{_fmt(result['world2base'])}")

    logger.log_info(f"\nPosition error (score): {result['score']:.4f}")
    logger.log_info(f"Rotation error: {result['rotation_score']:.4f} deg")
    logger.log_info(f"Frames used:     {len(result['valid_ids'])}")
    logger.log_info(f"Frames rejected: {result['n_outliers_rejected']} (indices: {rejected_ids})")

    logger.log_info("\nPer-frame:")
    frame_results = result["frame_results"]
    for k, (idx, fname) in enumerate(info):
        err = frame_results[k] if k < len(frame_results) else -1.0
        logger.log_info(f"  [{idx:3d}] err={err:.3f}  {fname}")

    out = os.path.join(args.output_dir, "calibration_result.npz")
    np.savez(out, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    logger.log_info(f"\nSaved → {out}")

    if hasattr(estimator, "corners_2d_list") and estimator.corners_2d_list:
        try:
            estimator.project_marker_points_to_image(
                os.path.join(args.output_dir, "projected_corners_2d.jpg")
            )
        except Exception as e:
            logger.log_warn(f"Failed to project marker points: {e}")

    # Optional: compare against a reference result.
    ref_path = getattr(args, "reference", None)
    if ref_path and os.path.isdir(ref_path):
        ref_path = os.path.join(ref_path, "calibration_result.npz")
    if ref_path and os.path.isfile(ref_path):
        ref = np.load(ref_path, allow_pickle=False)
        logger.log_info("\n=======================================")
        logger.log_info("COMPARISON vs REFERENCE")
        logger.log_info("=======================================")
        for key in ("flange2cam", "base2world"):
            T_cur = result.get(key)
            if key not in ref.files:
                continue
            T_ref = ref[key]
            if T_cur is None or T_ref is None:
                continue
            logger.log_info(f"current {key}:\n{_fmt(T_cur)}")
            logger.log_info(f"reference {key}:\n{_fmt(T_ref)}")
            d_mm = pose_translation_distance(T_cur, T_ref)
            d_deg = pose_angular_distance_deg(T_cur, T_ref)
            logger.log_info(f"{key}: t_dist={d_mm:.4f} mm, r_dist={d_deg:.4f} deg")
        ref.close()

    return result


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hand-eye (eye-in-hand) calibration — standalone calib_module version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
File naming convention:
  {index}_{tx}_{ty}_{tz}_{rx}_{ry}_{rz}.png  (RGB mode)
  {index}_{tx}_{ty}_{tz}_{rx}_{ry}_{rz}.ply  (PLY mode)

Example (PLY mode):
  python hand_eye_calibration.py \\
      --data_dir ./images \\
      --rows 8 --cols 7 --size 30 \\
      --input_type ply \\
      --rot_type euler --rot_unit deg --pos_unit mm

Example (RGB mode; *_intrinsic.json applied automatically if present):
  python hand_eye_calibration.py \\
      --data_dir ./images --input_type rgb
""")
    p.add_argument("--data_dir", default=USER_DEFAULT_DATA_DIR)
    p.add_argument("--output_dir", default=USER_DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--input_type", default=USER_DEFAULT_INPUT_TYPE,
        choices=["rgb", "ply", "rgb_depth"],
        help="Input type",
    )
    p.add_argument("--rows", type=int, default=USER_DEFAULT_ROWS)
    p.add_argument("--cols", type=int, default=USER_DEFAULT_COLS)
    p.add_argument("--size", type=float, default=USER_DEFAULT_SIZE, help="Checkerboard square size (mm)")
    p.add_argument("--rot_type", default=USER_DEFAULT_ROT_TYPE, choices=["euler", "rotvec", "quat"])
    p.add_argument("--rot_unit", default=USER_DEFAULT_ROT_UNIT, choices=["deg", "rad"])
    p.add_argument("--pos_unit", default=USER_DEFAULT_POS_UNIT, choices=["mm", "m"])
    p.add_argument("--euler_conv", default=USER_DEFAULT_EULER_CONV)
    _debug_group = p.add_mutually_exclusive_group()
    _debug_group.add_argument("--debug", dest="debug", action="store_true", default=USER_DEFAULT_DEBUG)
    _debug_group.add_argument("--no-debug", dest="debug", action="store_false")
    p.add_argument(
        "--intrinsic",
        nargs=4, type=float,
        metavar=("FX", "FY", "CX", "CY"),
        default=USER_DEFAULT_INTRINSIC,
        help="Camera intrinsic parameters (fx fy cx cy). Used when *_intrinsic.json is absent.",
    )
    p.add_argument(
        "--dist_coeffs",
        nargs="*", type=float,
        default=USER_DEFAULT_DIST_COEFFS,
        help="Distortion coefficients (OpenCV order). Defaults to 5 zeros if omitted.",
    )
    p.add_argument(
        "--reference",
        type=str, default=USER_DEFAULT_REFERENCE,
        metavar="NPZ",
        help="Path to a reference .npz file or directory (must contain flange2cam and base2world)",
    )

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
