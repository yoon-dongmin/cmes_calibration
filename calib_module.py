"""
calib_module.py

Standalone module that replicates crp_calibration / crp_core functionality
without requiring those packages to be installed.

Contents:
  - Logger           : console + file logging (replaces crp_core.module_base.logger)
  - Transform utils  : inv_se3, get_se3, rotation_to_matrix, etc.
  - IO utils         : read_ply_file, save_image_file, etc.
  - ErrorCode / CalibrationError
  - Frame / Frames
  - marker_3d_pose_pnp, CameraEstimator
  - FixedEyeCalibration
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os
import re
import json
import copy
import logging
import threading
import itertools
import colorsys
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# ─── Third-party ─────────────────────────────────────────────────────────────
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as Rot


# ═════════════════════════════════════════════════════════════════════════════
#  Logger  (replaces crp_core.module_base.logger)
# ═════════════════════════════════════════════════════════════════════════════

class Logger:
    """
    Lightweight logger with the same interface as the crp_core Logger.
    Outputs to both the console and a date-stamped log file.
    """

    def __init__(self, name: str, project_name: str = "", log_path: str = "./logs"):
        self.name = name
        self.project_name = project_name
        self.log_path = log_path
        self.debug_mode: bool = False

        self._logger = logging.getLogger(f"{project_name}.{name}" if project_name else name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if not self._logger.handlers:
            fmt = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

            # File handler
            if log_path:
                now = datetime.now()
                log_dir = Path(log_path) / project_name / "logs" / now.strftime("%Y-%m")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"log_{now.strftime('%Y%m%d')}.log"
                fh = logging.FileHandler(str(log_file), encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(fmt)
                self._logger.addHandler(fh)

    def log_info(self, msg: str) -> None:
        self._logger.info(msg)

    def log_warn(self, msg: str) -> None:
        self._logger.warning(msg)

    def log_error(self, msg: str) -> None:
        self._logger.error(msg)

    def log_debug(self, msg: str) -> None:
        self._logger.debug(msg)


# ═════════════════════════════════════════════════════════════════════════════
#  Transform Utilities  (crp_calibration.utils_3d.transform_utils)
# ═════════════════════════════════════════════════════════════════════════════

def _is_rotation_matrix(M: np.ndarray, atol: float = 1e-3) -> bool:
    if not isinstance(M, np.ndarray) or M.shape != (3, 3):
        return False
    should_be_I = M.T @ M
    return np.allclose(should_be_I, np.eye(3), atol=atol) and np.isclose(
        np.linalg.det(M), 1.0, atol=atol
    )


def inv_se3(se3_matrix: np.ndarray) -> np.ndarray:
    """Numerically stable inverse of a 4×4 SE3 matrix using R^T."""
    if not isinstance(se3_matrix, np.ndarray) or se3_matrix.shape != (4, 4):
        raise ValueError("inv_se3: Input must be a 4x4 numpy array.")
    R_part = se3_matrix[:3, :3]
    t_part = se3_matrix[:3, 3]
    if not _is_rotation_matrix(R_part):
        raise ValueError("inv_se3: Upper-left 3x3 is not a valid rotation matrix.")
    R_inv = R_part.T
    t_inv = -np.dot(R_inv, t_part)
    out = np.eye(4)
    out[:3, :3] = R_inv
    out[:3, 3] = t_inv
    return out


def inverse_transform(
    rotation_matrix: np.ndarray, translation_vector: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the inverse of a rotation matrix and translation vector pair."""
    if not _is_rotation_matrix(rotation_matrix):
        raise ValueError("inverse_transform: rotation_matrix must be a valid 3×3 rotation.")
    t = np.asarray(translation_vector, dtype=np.float64).reshape(-1)
    if t.shape != (3,):
        raise ValueError("inverse_transform: translation_vector must have shape (3,).")
    inv_R = rotation_matrix.T
    inv_t = -inv_R @ t
    return inv_R, inv_t


def rotation_vector_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    return Rot.from_rotvec([rx, ry, rz]).as_matrix()


def euler_to_matrix(
    rx: float, ry: float, rz: float, *, degrees: bool = False
) -> np.ndarray:
    return Rot.from_euler("xyz", [rx, ry, rz], degrees=degrees).as_matrix()


def static_xyz_to_matrix(w: float, p: float, r: float) -> np.ndarray:
    return Rot.from_euler("XYZ", [w, p, r]).as_matrix()


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return Rot.from_quat([x, y, z, w]).as_matrix()


def rotation_to_matrix(
    type: str, data, *, degrees: bool = False
) -> np.ndarray:
    """Convert a rotation representation (rotvec / euler / staticXYZ / quat) to a 3×3 rotation matrix."""
    t = (type or "").strip().lower()
    if isinstance(data, dict):
        keys = {k.lower(): v for k, v in data.items()}
        if t == "quat" and all(k in keys for k in ("x", "y", "z", "w")):
            data = [keys["x"], keys["y"], keys["z"], keys["w"]]
        elif t in ("rotvec", "euler") and all(k in keys for k in ("x", "y", "z")):
            data = [keys["x"], keys["y"], keys["z"]]
        elif type == "staticXYZ" and all(k in data for k in ("x", "y", "z")):
            data = [data["x"], data["y"], data["z"]]
        else:
            raise ValueError(f"rotation_to_matrix: invalid dict keys for type '{type}': {data}")

    if not isinstance(data, (list, tuple, np.ndarray)):
        raise ValueError("rotation_to_matrix: data must be a list/tuple/ndarray.")

    if t == "rotvec" and len(data) == 3:
        return rotation_vector_to_matrix(*data)
    elif t == "euler" and len(data) == 3:
        return euler_to_matrix(*data, degrees=degrees)
    elif type == "staticXYZ" and len(data) == 3:
        return static_xyz_to_matrix(*data)
    elif t == "quat" and len(data) == 4:
        return quaternion_to_matrix(*data)
    else:
        raise ValueError(
            f"rotation_to_matrix: invalid type or data length. type={type}, data={data}"
        )


def get_se3(robot_R: np.ndarray, robot_t) -> np.ndarray:
    """R(3×3) + t → SE3 4×4."""
    if not isinstance(robot_R, np.ndarray) or robot_R.shape != (3, 3):
        raise ValueError("robot_R must be a 3×3 numpy array.")

    if isinstance(robot_t, dict):
        if all(key in robot_t for key in ("x", "y", "z")):
            robot_t = [robot_t["x"], robot_t["y"], robot_t["z"]]
        else:
            raise ValueError("robot_t dict must contain 'x', 'y', 'z' keys.")
    robot_t = np.asarray(robot_t, dtype=np.float64).reshape(-1)
    if robot_t.shape[0] == 1 or robot_t.ndim != 1:
        robot_t = robot_t.flatten()
    if robot_t.shape != (3,):
        raise ValueError(f"robot_t must have shape (3,), got {robot_t.shape}")

    se3 = np.eye(4)
    se3[:3, :3] = robot_R
    se3[:3, 3] = robot_t
    return se3


# ═════════════════════════════════════════════════════════════════════════════
#  IO Utilities  (crp_calibration.utils_3d.io_utils)
# ═════════════════════════════════════════════════════════════════════════════

_DEFAULTS_LOCK = threading.RLock()
_DEFAULT_PLY_PREFIX: str = ""
_DEFAULT_DEBUG: str = "debug"
_SAFE_PREFIX_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_prefix(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return _SAFE_PREFIX_RE.sub("_", s)[:128] if s else ""


def set_ply_prefix(prefix: Optional[str]) -> None:
    global _DEFAULT_PLY_PREFIX
    with _DEFAULTS_LOCK:
        _DEFAULT_PLY_PREFIX = _sanitize_prefix(prefix or "")


def get_ply_prefix() -> str:
    with _DEFAULTS_LOCK:
        return _DEFAULT_PLY_PREFIX


def set_debug_path(path) -> None:
    global _DEFAULT_DEBUG
    norm = str(Path(path)).replace("\\", "/")
    if not norm:
        raise ValueError("debug path cannot be empty")
    with _DEFAULTS_LOCK:
        _DEFAULT_DEBUG = norm


def read_ply_file(filename, include_attrs: bool = False):
    """Read a PLY or STL file and return a PointCloud or TriangleMesh."""
    if isinstance(filename, (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh)):
        return filename

    path = Path(filename).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".ply":
            try:
                tpcd = o3d.t.io.read_point_cloud(str(path))
                if tpcd.is_empty():
                    raise ValueError(f"PLY '{path}' is empty.")
            except Exception:
                pcd = o3d.io.read_point_cloud(str(path))
                if pcd.is_empty():
                    raise ValueError(f"PLY '{path}' is empty.")
                return (pcd, {}) if include_attrs else pcd

            attrs = {}
            for k, v in tpcd.point.items():
                if k in ("positions", "colors", "normals"):
                    continue
                arr = v.numpy()
                attrs[k] = arr.reshape(-1) if arr.ndim == 2 and arr.shape[1] == 1 else arr

            pcd = tpcd.to_legacy()
            return (pcd, attrs) if include_attrs else pcd

        elif suffix == ".stl":
            mesh = o3d.io.read_triangle_mesh(str(path))
            if not mesh.has_vertices():
                raise ValueError(f"STL '{path}' has no vertices.")
            return mesh
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    except Exception as e:
        raise IOError(f"Failed to read '{path}': {e}") from e


def save_image_file(filename, image, use_default: bool = False, colorspace: str = "BGR"):
    """Save an image to a file with Unicode path support."""
    directory, base_filename = os.path.split(filename)
    if not directory:
        directory = "."

    root, ext = os.path.splitext(base_filename)
    if not ext:
        ext = ".png"
        base_filename = root + ext

    if use_default:
        target_dir = os.path.join(directory, _DEFAULT_DEBUG)
        target_name = (
            f"{_DEFAULT_PLY_PREFIX}_{base_filename}" if _DEFAULT_PLY_PREFIX else base_filename
        )
        filename = os.path.join(target_dir, target_name)
    else:
        filename = os.path.join(
            directory,
            f"{_DEFAULT_PLY_PREFIX}_{base_filename}" if _DEFAULT_PLY_PREFIX else base_filename,
        )

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    if image is None:
        raise ValueError("Image is None.")

    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        raise IOError(f"OpenCV imencode failed for '{filename}'")
    with open(filename, "wb") as f:
        f.write(buf.tobytes())
    return filename


def read_json_file(filename, unit: str = "m") -> dict:
    """Read a JSON file and convert position data to the specified unit."""
    conversion = {"m": 1000, "cm": 10, "mm": 1}
    with open(filename, "r") as f:
        data = json.load(f)
    factor = conversion.get(unit, 1)
    if "position" in data:
        data["position"] = {k: v * factor for k, v in data["position"].items()}
    return data


def save_ply_file(filename, pcd, use_default: bool = False, write_ascii: bool = False):
    """Save a PointCloud or NumPy array as a PLY file."""
    directory, base_filename = os.path.split(filename)
    if not directory:
        directory = "."
    if not base_filename.lower().endswith(".ply"):
        base_filename += ".ply"

    if use_default:
        target_dir = os.path.join(directory, _DEFAULT_DEBUG)
        target_name = (
            f"{_DEFAULT_PLY_PREFIX}_{base_filename}" if _DEFAULT_PLY_PREFIX else base_filename
        )
        filename = os.path.join(target_dir, target_name)
    else:
        filename = os.path.join(
            directory,
            f"{_DEFAULT_PLY_PREFIX}_{base_filename}" if _DEFAULT_PLY_PREFIX else base_filename,
        )

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    if isinstance(pcd, np.ndarray):
        if pcd.ndim == 1:
            pcd = pcd.reshape(1, -1)
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3].astype(np.float64))
        if pcd.shape[1] >= 6:
            cols = pcd[:, 3:6].astype(np.float64)
            if cols.max() > 1.0:
                cols = cols / 255.0
            o3d_pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
        pcd = o3d_pcd

    if isinstance(pcd, o3d.geometry.PointCloud):
        ok = o3d.io.write_point_cloud(filename, pcd, write_ascii=write_ascii, print_progress=False)
        if not ok:
            raise IOError(f"Failed to save PLY: {filename}")
        return filename

    raise TypeError("Input must be Open3D PointCloud or NumPy array.")


# ═════════════════════════════════════════════════════════════════════════════
#  Error Handler  (crp_calibration.module.error_handler)
# ═════════════════════════════════════════════════════════════════════════════

MODULE_NAME = "008"


class ErrorSeverity(Enum):
    Trivial = "0"
    Normal = "1"
    Warning = "2"
    Error = "3"
    Critical = "4"


class ErrorCode(Enum):
    UNKNOWN = (0, "Unknown cause of error")
    NOT_INITIALIZED = (1, "Module not initialized")
    INVALID_MODE = (2, "Invalid operation mode selected")
    FACTORY_CREATION_FAILED = (3, "Factory creation failed")
    FACTORY_CREATION_EXCEPTION = (4, "Exception during factory creation")
    SYSTEM_RESET_FAILED = (101, "Module reset failed")
    EMPTY_FILE = (201, "File is empty")
    FILE_LOAD_FAILED = (202, "Failed to load file")
    FILE_NOT_FOUND = (203, "File not found")
    INSUFFICIENT_FRAMES = (401, "Not enough frames available")
    ESTIMATOR_POSE_FAILED = (402, "Pose estimation failed")
    NO_MARKER_FOUND = (403, "Marker not detected")
    INVALID_FILE_STRUCTURE = (404, "Invalid file structure")
    INVALID_PARAM = (405, "Invalid parameter")
    SAVE_FAILED = (406, "Failed to save")
    EXECUTION_FAILED = (501, "Error during execution")
    FRAME_SAVE_FAILED = (502, "Failed to save frame")
    TRANSFORM_FAILED = (503, "Transformation failed")
    FRAME_ID_MISMATCH = (504, "Frame ID mismatch")

    def __init__(self, num, default_msg):
        self.num = num
        self.default_msg = default_msg

    def code_str(self, severity):
        mod_raw = str(MODULE_NAME)
        mod_digits = "".join(ch for ch in mod_raw if ch.isdigit()) or "0"
        mod3 = mod_digits[-3:].zfill(3)
        mod_val = int(mod3)
        try:
            sev = abs(int(severity)) % 10
        except Exception:
            sev = 0
        num = abs(int(self.num)) % 10000
        code = mod_val * 100000 + sev * 10000 + num
        return max(0, min(code, 0xFFFFFFFF))

    @property
    def default_message(self):
        return self.default_msg


class CalibrationError(Exception):
    def __init__(self, error_code: ErrorCode, severity: str, message: Optional[str] = None):
        self.error_code = error_code
        self.severity = severity
        self.message = message or error_code.default_message
        self.code = error_code.code_str(severity)
        super().__init__(self.message)

    def to_msg(self) -> str:
        return f"[{self.error_code.code_str(self.severity)}] {self.message}"


# ═════════════════════════════════════════════════════════════════════════════
#  Frames  (crp_calibration.core.frames)
# ═════════════════════════════════════════════════════════════════════════════

class Frame:
    def __init__(self, frame_id, robot=None, camera=None, points=None):
        self.frame_id = frame_id
        self.robot_pose = self._validate_pose(robot)
        self.camera_pose = self._validate_pose(camera)
        self.camera_points = points

    @staticmethod
    def _validate_pose(pose):
        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
            return pose
        raise ValueError("Invalid pose data. Must be a 4×4 numpy array.")

    def add_data(self, key, value):
        if not hasattr(self, "_extra"):
            self._extra = {}
        self._extra[key] = value

    def get_data(self, key):
        return getattr(self, "_extra", {}).get(key)


class Frames:
    def __init__(self, logger=None):
        self.frames: Dict[Any, Frame] = {}
        self.logger = logger

    def add_frame(self, frame: Frame) -> None:
        self.frames[frame.frame_id] = frame

    def get_frame(self, frame_id) -> Optional[Frame]:
        return self.frames.get(frame_id)

    def remove_frame(self, frame_id) -> None:
        if frame_id in self.frames:
            del self.frames[frame_id]
        else:
            raise KeyError(f"Frame ID {frame_id} does not exist.")

    def get_all_frames(self) -> List[Frame]:
        return list(self.frames.values())

    def items(self):
        return self.frames.items()

    def clear(self) -> None:
        self.frames.clear()

    def save_to_file(self, file_path: str) -> None:
        data = {
            fid: {
                "robot_pose": frame.robot_pose.tolist(),
                "camera_pose": frame.camera_pose.tolist(),
            }
            for fid, frame in self.frames.items()
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def load_from_file(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)
        self.frames = {
            fid: Frame(
                fid,
                robot=np.array(v["robot_pose"]),
                camera=np.array(v["camera_pose"]),
            )
            for fid, v in data.items()
        }

    def __getitem__(self, key):
        return self.frames[key]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)


# ═════════════════════════════════════════════════════════════════════════════
#  Estimator utilities  (crp_calibration.core.estimator — shared helpers)
# ═════════════════════════════════════════════════════════════════════════════

def _pca_straightness(P: np.ndarray) -> float:
    C = P - P.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(C, full_matrices=False)
    if S.size < 2 or (S[0] + S[1]) < 1e-9:
        return 0.0
    return float(np.clip(1.0 - (S[1] / (S[0] + 1e-9)), 0.0, 1.0))


def _fitline_rmse_norm(P: np.ndarray) -> float:
    P32 = P.astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(P32, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    v = np.array([vx, vy], dtype=np.float64)
    v /= np.linalg.norm(v) + 1e-9
    p0 = np.array([x0, y0], dtype=np.float64)
    v_perp = np.array([-v[1], v[0]])
    d = np.abs((P - p0) @ v_perp)
    rmse = float(np.sqrt(np.mean(d ** 2)))
    spacing = np.mean(np.linalg.norm(P[1:] - P[:-1], axis=1)) if len(P) >= 2 else 1.0
    return rmse / (spacing + 1e-9)


def marker_3d_pose(
    camera_points: np.ndarray, target_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD-based rigid-body transform estimation: target → camera frame."""
    camera_points = np.asarray(camera_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if camera_points.shape != target_points.shape or camera_points.size == 0:
        raise ValueError("Points must have identical non-empty shapes.")

    cc = np.mean(camera_points, axis=0)
    ct = np.mean(target_points, axis=0)
    H = (target_points - ct).T @ (camera_points - cc)
    U, _, Vt = np.linalg.svd(H)
    R_c2t = Vt.T @ U.T
    if np.all(np.isfinite(R_c2t)) and np.linalg.det(R_c2t) < 0:
        Vt[-1, :] *= -1
        R_c2t = Vt.T @ U.T
    t_c2t = cc - R_c2t @ ct
    R_t2c, t_t2c = inverse_transform(R_c2t, t_c2t)
    return R_t2c, t_t2c


def marker_3d_pose_pnp(
    corners_2d: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    PnP-based pose estimation (solvePnP + LM refinement).

    Returns: (R_target2cam 3×3, t_target2cam (3,), reproj_error_px)
    """
    corners_2d = np.asarray(corners_2d, dtype=np.float64).reshape(-1, 1, 2)
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points, corners_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise CalibrationError(ErrorCode.ESTIMATOR_POSE_FAILED, "3", "solvePnP failed.")

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points, corners_2d, camera_matrix, dist_coeffs, rvec, tvec
    )

    proj_pts, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_error = float(
        np.mean(np.linalg.norm(proj_pts.reshape(-1, 2) - corners_2d.reshape(-1, 2), axis=1))
    )

    R_t2c, _ = cv2.Rodrigues(rvec)
    return R_t2c, tvec.flatten(), reproj_error


def _flatten_pixels(pixel_2d: np.ndarray):
    pixel_2d = np.asarray(pixel_2d, dtype=float)
    if pixel_2d.ndim == 3 and pixel_2d.shape[-2:] == (1, 2):
        pixel_2d = pixel_2d.reshape(pixel_2d.shape[0], 2)
    if pixel_2d.ndim < 2 or pixel_2d.shape[-1] != 2:
        raise ValueError(f"pixel_2d must have shape (...,2). got {pixel_2d.shape}")
    leading_shape = pixel_2d.shape[:-1]
    return pixel_2d.reshape(-1, 2), leading_shape


def find_3d_points_from_2d(
    intrinsic_matrix: np.ndarray,
    pcd: o3d.geometry.PointCloud,
    pixel_2d: np.ndarray,
    k: int = 12,
    method: str = "plane",
    depth_gate: Optional[float] = 10.0,
    idw_power: float = 2.0,
    gaussian_sigma_px: Optional[float] = 2.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """Lift 2D pixel coordinates to 3D points using KD-tree nearest neighbours + plane fitting."""
    flat_uv, leading_shape = _flatten_pixels(pixel_2d)

    fx, fy = float(intrinsic_matrix[0, 0]), float(intrinsic_matrix[1, 1])
    cx, cy = float(intrinsic_matrix[0, 2]), float(intrinsic_matrix[1, 2])

    points = np.asarray(pcd.points, dtype=float)
    valid_mask = points[:, 2] > 0
    P = points[valid_mask]

    z = P[:, 2]
    u = fx * P[:, 0] / z + cx
    v = fy * P[:, 1] / z + cy
    proj = np.column_stack((u, v))

    tree = KDTree(proj)
    k_eff = min(k, proj.shape[0])
    dists, idxs = tree.query(flat_uv, k=k_eff)
    if k_eff == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    out = np.full((flat_uv.shape[0], 3), np.nan, dtype=float)

    for i in range(flat_uv.shape[0]):
        uv = flat_uv[i]
        di = dists[i].astype(float)
        ii = idxs[i].astype(int)
        neigh_P = P[ii]
        neigh_d = di

        if depth_gate is not None and neigh_P.shape[0] >= 3:
            z_med = np.median(neigh_P[:, 2])
            keep = np.abs(neigh_P[:, 2] - z_med) <= float(depth_gate)
            neigh_P = neigh_P[keep]
            neigh_d = neigh_d[keep]

        if neigh_P.shape[0] < 3:
            w = 1.0 / (neigh_d + eps) ** idw_power
            w /= w.sum() + eps
            out[i] = (w[:, None] * neigh_P).sum(axis=0)
            continue

        if method == "plane":
            C = neigh_P.mean(axis=0)
            _, _, Vt2 = np.linalg.svd(neigh_P - C, full_matrices=False)
            n = Vt2[-1]
            nn = np.linalg.norm(n)
            if nn >= 1e-12:
                n /= nn
                d0 = -np.dot(n, C)
                ray = np.array([(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0])
                denom = np.dot(n, ray)
                if abs(denom) >= 1e-9:
                    t_val = -d0 / denom
                    if t_val > 0:
                        out[i] = t_val * ray
                        continue

        # IDW fallback
        if gaussian_sigma_px is not None:
            w = np.exp(-(neigh_d ** 2) / (2.0 * (gaussian_sigma_px ** 2) + eps))
        else:
            w = 1.0 / (neigh_d + eps) ** idw_power
        w /= w.sum() + eps
        out[i] = (w[:, None] * neigh_P).sum(axis=0)

    return out.reshape(*leading_shape, 3)


# ═════════════════════════════════════════════════════════════════════════════
#  Estimator Base  (crp_calibration.core.estimator.Estimator)
# ═════════════════════════════════════════════════════════════════════════════

class Estimator:
    def __init__(self, logger: Logger):
        self.marker_info: dict = {}
        self.logger = logger
        self.result = None
        self.search_radius = 2

        self.width, self.height = 2000, 2000
        self.original_width = self.width
        self.original_height = self.height

        fx = fy = self.width
        cx = self.width / 2
        cy = self.height / 2
        self.intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0.0, 0.0, 1.0]])
        self.original_intrinsic_matrix = self.intrinsic_matrix.copy()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def set_scale_matrix(self, scale_x: float, scale_y: float) -> None:
        if scale_x == 1.0 and scale_y == 1.0:
            return
        if scale_x <= 0 or scale_y <= 0:
            self.logger.log_warn(f"Invalid scale: ({scale_x}, {scale_y}), using (1.0, 1.0)")
            return
        new_width = int(self.original_width * scale_x)
        new_height = int(self.original_height * scale_y)
        if new_width <= 0 or new_height <= 0:
            self.logger.log_warn(f"Invalid scaled dims: {new_width}×{new_height}, skipping")
            return
        self.width = new_width
        self.height = new_height
        self.intrinsic_matrix = self.original_intrinsic_matrix.copy()
        self.intrinsic_matrix[0, 0] *= scale_x
        self.intrinsic_matrix[1, 1] *= scale_y
        self.intrinsic_matrix[0, 2] *= scale_x
        self.intrinsic_matrix[1, 2] *= scale_y
        self.logger.log_debug(f"Scaled intrinsic:\n{self.intrinsic_matrix}")

    def _point_cloud_to_rgb_image(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        if not pcd.has_points() or not pcd.has_colors():
            raise CalibrationError(
                ErrorCode.FILE_LOAD_FAILED, "3",
                "Point cloud must have both points and colors.",
            )
        pts = np.asarray(pcd.points)
        cols = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        K = self.intrinsic_matrix
        z = pts[:, 2]
        valid = z != 0
        pts, cols, z = pts[valid], cols[valid], z[valid]

        u = ((K[0, 0] * pts[:, 0] / z) + K[0, 2]).astype(int)
        v = ((K[1, 1] * pts[:, 1] / z) + K[1, 2]).astype(int)

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        m = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        img[v[m], u[m]] = cols[m]
        return img

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _gamma(img: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / float(gamma)
        lut = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)

    @staticmethod
    def _illumination_correction(img: np.ndarray, sigma: float = 11) -> np.ndarray:
        base = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.divide(img, base, scale=255)

    @staticmethod
    def _specular_inpaint(img: np.ndarray, thr: int = 245) -> np.ndarray:
        _, m = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(m) == 0:
            return img
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _unsharp(img: np.ndarray, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

    @staticmethod
    def _guided(img: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
        try:
            gf = cv2.ximgproc.guidedFilter(guide=img, src=img, radius=radius, eps=eps)
            return Estimator._to_uint8(gf)
        except Exception:
            return cv2.bilateralFilter(img, d=radius * 2 + 1, sigmaColor=50, sigmaSpace=radius)

    def prepare_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image.copy()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def filter_variants(self, gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        c = self._clahe
        return [("clahe", c.apply(gray))]

    def _adjust_scale(self, scale_factor: float, reason: str) -> None:
        cx = self.width / self.original_width
        cy_s = self.height / self.original_height
        new_sx = round(cx * scale_factor, 3)
        new_sy = round(cy_s * scale_factor, 3)
        self.logger.log_info(
            f"{reason}, scale: ({cx:.3f},{cy_s:.3f})→({new_sx},{new_sy})"
        )
        self.set_scale_matrix(new_sx, new_sy)
        self.param["cam_scale_x"] = new_sx
        self.param["cam_scale_y"] = new_sy

    def _load_and_parse_files(
        self, file_list, rot_type, rot_unit, scale_unit, input_type="ply"
    ):
        """Load PLY / image / JSON data from a file list and parse the robot pose."""
        base_pose = np.eye(4)
        ply_data = rgb_image = depth_map = None
        last_file = ""
        RGB_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
        DEPTH_EXTS = {".tif", ".tiff"}

        try:
            for file in file_list:
                if file is None:
                    continue
                ext = os.path.splitext(file)[1].lower()
                last_file = file

                if ext == ".ply" and input_type in ("ply", "rgb_depth"):
                    try:
                        ply_data = read_ply_file(file)
                        self.logger.log_info(f"Read PLY: {file}")
                    except Exception as e:
                        raise CalibrationError(
                            ErrorCode.FILE_LOAD_FAILED, "3", f"PLY read failed: {file}, {e}"
                        )

                elif ext == ".json":
                    try:
                        json_data = read_json_file(file, unit="m")
                        self.logger.log_info(f"Read JSON: {file}")
                    except Exception as e:
                        raise CalibrationError(
                            ErrorCode.FILE_LOAD_FAILED, "3", f"JSON read failed: {file}, {e}"
                        )
                    try:
                        base_pose = get_se3(
                            rotation_to_matrix("quat", json_data["orientation"]),
                            json_data["position"],
                        )
                    except Exception as e:
                        raise CalibrationError(
                            ErrorCode.FILE_LOAD_FAILED, "3", f"Invalid JSON pose: {file}, {e}"
                        )

                elif ext in RGB_EXTS and input_type in ("rgb", "rgb_depth"):
                    try:
                        with open(file, "rb") as f:
                            buf = np.frombuffer(f.read(), dtype=np.uint8)
                        rgb_image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if rgb_image is None:
                            raise ValueError("imdecode returned None")
                        self.logger.log_info(f"Read Image: {file}")
                    except Exception as e:
                        raise CalibrationError(
                            ErrorCode.FILE_LOAD_FAILED, "3", f"Image read failed: {file}, {e}"
                        )

                elif ext in DEPTH_EXTS and input_type == "rgb_depth":
                    try:
                        depth_map = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                        if depth_map is None:
                            raise ValueError("imread returned None for depth")
                        depth_map = depth_map.astype(np.float32)
                        self.logger.log_info(f"Read Depth: {file}")
                    except Exception as e:
                        raise CalibrationError(
                            ErrorCode.FILE_LOAD_FAILED, "3", f"Depth read failed: {file}, {e}"
                        )

            # Validate required data for each input type.
            if input_type == "ply" and ply_data is None:
                raise CalibrationError(ErrorCode.FILE_LOAD_FAILED, "3", "No PLY file (input_type='ply').")
            if input_type == "rgb" and rgb_image is None:
                raise CalibrationError(ErrorCode.FILE_LOAD_FAILED, "3", "No image file (input_type='rgb').")
            if input_type == "rgb_depth":
                if rgb_image is None:
                    raise CalibrationError(ErrorCode.FILE_LOAD_FAILED, "3", "No image file (rgb_depth).")
                if ply_data is None:
                    raise CalibrationError(ErrorCode.FILE_LOAD_FAILED, "3", "No PLY file (rgb_depth).")

            # Fall back to parsing pose from the filename when no JSON is present.
            if np.abs(base_pose - np.eye(4)).max() == 0:
                name_part = os.path.basename(last_file).rsplit(".", 1)[0].split("_")
                try:
                    cam_rot = [float(name_part[4]), float(name_part[5]), float(name_part[6])]
                    if rot_unit == "deg":
                        cam_rot = [np.deg2rad(r) for r in cam_rot]
                    cam_R = rotation_to_matrix(rot_type, cam_rot)
                    cam_t = [float(name_part[1]), float(name_part[2]), float(name_part[3])]
                except Exception as e:
                    raise CalibrationError(
                        ErrorCode.FILE_LOAD_FAILED, "3",
                        f"Filename parse failed: {last_file}, {e}"
                    )
                base_pose = get_se3(cam_R, cam_t)
                if scale_unit == "m":
                    base_pose[:3, 3] *= 1000

            return ply_data, base_pose, rgb_image, depth_map

        except CalibrationError:
            raise
        except Exception as e:
            raise CalibrationError(ErrorCode.FILE_LOAD_FAILED, "3", f"Unknown error: {e}")

    def get_pose(self, *args, **kwargs):
        raise NotImplementedError

    def project_marker_points_to_image(self, save_path: Optional[str] = None) -> None:
        if not self.corners_2d_list:
            self.logger.log_warn("No corners_2d data to project")
            return

        pts_chunks = [
            np.asarray(c, dtype=np.float64).reshape(-1, 2)
            for c in self.corners_2d_list
            if c is not None and len(c) > 0
        ]
        if not pts_chunks:
            return

        P = np.vstack(pts_chunks)
        margin = 40
        img_w = max(int(np.ceil(P[:, 0].max()) + margin), 320)
        img_h = max(int(np.ceil(P[:, 1].max()) + margin), 240)
        proj_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        ]
        for frame_idx, (corners_2d, frame_id) in enumerate(
            zip(self.corners_2d_list, self.frame_ids_list)
        ):
            if corners_2d is None or len(corners_2d) == 0:
                continue
            color = colors[frame_idx % len(colors)]
            ci = corners_2d.astype(int)
            valid = (
                (ci[:, 0] >= 0) & (ci[:, 0] < img_w) &
                (ci[:, 1] >= 0) & (ci[:, 1] < img_h)
            )
            for corner in ci[valid]:
                cv2.circle(proj_image, tuple(corner), 3, color, -1)
            if ci[valid].shape[0] > 0:
                tx, ty = ci[valid][0]
                cv2.putText(
                    proj_image, f"F{frame_id}", (tx, ty - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

        if save_path is None:
            save_path = "projected_corners_2d.jpg"
        cv2.imwrite(save_path, proj_image)
        self.logger.log_info(
            f"Projected {len(self.corners_2d_list)} frames' corners_2d → {save_path}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  CameraEstimator  (crp_calibration.core.estimator.CameraEstimator)
# ═════════════════════════════════════════════════════════════════════════════

class CameraEstimator(Estimator):
    def __init__(self, logger: Logger, param: dict):
        super().__init__(logger)
        self.param = param
        self.pattern_size = (param["rows"] - 1, param["cols"] - 1)
        self.corners_2d_list: List = []
        self.frame_ids_list: List = []
        self.last_mean_corner_spacing: Optional[float] = None

        self.real_camera_matrix: Optional[np.ndarray] = None
        self.real_dist_coeffs: Optional[np.ndarray] = None
        if "camera_matrix" in param:
            self.real_camera_matrix = np.array(param["camera_matrix"], dtype=np.float64).reshape(3, 3)
        if "dist_coeffs" in param:
            self.real_dist_coeffs = np.array(param["dist_coeffs"], dtype=np.float64)

        if param.get("ply_render_use_real_camera") and self.real_camera_matrix is not None:
            self.intrinsic_matrix = self.real_camera_matrix.copy()
            self.original_intrinsic_matrix = self.intrinsic_matrix.copy()
            cx_r = float(self.real_camera_matrix[0, 2])
            cy_r = float(self.real_camera_matrix[1, 2])
            self.width = max(1, int(round(2 * cx_r)))
            self.height = max(1, int(round(2 * cy_r)))
            self.original_width = self.width
            self.original_height = self.height

        if "cam_scale_x" in param or "cam_scale_y" in param:
            super().set_scale_matrix(
                float(param.get("cam_scale_x", 1.0)),
                float(param.get("cam_scale_y", 1.0)),
            )

    def compute_2d_corner_spacing(self, corners_2d: np.ndarray) -> float:
        if len(corners_2d) < 2:
            return 0.0
        cb = (self.param["rows"] - 1, self.param["cols"] - 1)
        grid = corners_2d.reshape(cb[0], cb[1], 2)
        dists = []
        for r in range(cb[0]):
            for c in range(cb[1] - 1):
                dists.append(np.linalg.norm(grid[r, c + 1] - grid[r, c]))
        for r in range(cb[0] - 1):
            for c in range(cb[1]):
                dists.append(np.linalg.norm(grid[r + 1, c] - grid[r, c]))
        return float(np.mean(dists)) if dists else 0.0

    def _score(self, gray_u8: np.ndarray, corners: np.ndarray) -> float:
        Hc, Wc = self.pattern_size
        pts = corners.reshape(-1, 2).astype(np.float64)
        if pts.shape[0] != Hc * Wc:
            return 0.0
        grid = pts.reshape(Hc, Wc, 2)
        row_scores = [_pca_straightness(grid[r, :, :]) for r in range(Hc)]
        col_scores = [_pca_straightness(grid[:, c, :]) for c in range(Wc)]
        if not row_scores or not col_scores:
            return 0.0
        return float(np.clip(0.5 * (np.mean(row_scores) + np.mean(col_scores)), 0.0, 1.0))

    def _detect_single(self, img_u8: np.ndarray) -> Optional[np.ndarray]:
        flag_combinations = (
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS | cv2.CALIB_CB_FAST_CHECK,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
            0,
        )
        best_corners = None
        best_score = -1.0

        for img_try in (img_u8, cv2.bitwise_not(img_u8)):
            polarity_had_hit = False
            for flags in flag_combinations:
                ret, corners = cv2.findChessboardCorners(img_try, self.pattern_size, flags)
                if ret and corners is not None:
                    polarity_had_hit = True
                    corners_ref = cv2.cornerSubPix(
                        img_try, corners.copy(), (11, 11), (-1, -1),
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                    )
                    score = self._score(img_try, corners_ref)
                    if score > best_score:
                        best_score = score
                        best_corners = corners_ref
            if polarity_had_hit:
                break
        return best_corners

    def detect_checkerboard(self, image: np.ndarray):
        gray = self.prepare_gray(image)
        candidates = self.filter_variants(gray)
        best_corners, best_tag, best_score, best_img = None, "none", -1.0, None

        for tag, img in candidates:
            try:
                corners = self._detect_single(img)
                if corners is None:
                    continue
                score = self._score(img, corners)

                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, self.pattern_size, corners, True)
                txt = f"{tag} | s={score:.3f}"
                h, w = vis.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(0.5, min(1.2, (w + h) / 2000.0))
                thickness = max(1, int(2 * scale))
                (tw, th_t), bl = cv2.getTextSize(txt, font, scale, thickness)
                cv2.rectangle(vis, (5, 5), (5 + tw + 12, 5 + th_t + bl + 12), (0, 0, 0), -1)
                cv2.putText(vis, txt, (11, 11 + th_t), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)

                if self.logger.debug_mode:
                    save_image_file(f"{tag}_image.png", vis)

                if score > best_score:
                    best_score = score
                    best_corners = corners
                    best_tag = tag
                    best_img = img
            except Exception:
                continue

        if best_corners is None:
            self.logger.log_info("Checkerboard not found in all pipelines.")
            return None, "none", (0, 0.0), None

        vis = cv2.cvtColor(best_img, cv2.COLOR_GRAY2BGR)
        try:
            cv2.drawChessboardCorners(vis, self.pattern_size, best_corners, True)
        except Exception:
            pass

        n = len(best_corners)
        self.logger.log_info(
            f"Checkerboard detected (pipeline={best_tag}, n={n}, score={best_score:.3f})"
        )
        return best_corners, best_tag, (n, float(best_score)), vis

    def _build_object_points(self) -> np.ndarray:
        cb = (self.param["rows"] - 1, self.param["cols"] - 1)
        obj = np.zeros((cb[0] * cb[1], 3), np.float32)
        obj[:, :2] = np.mgrid[0 : cb[0], 0 : cb[1]].T.reshape(-1, 2)
        obj *= self.param["size"]
        return obj

    def estimate_pose_pnp(
        self,
        pcd: o3d.geometry.PointCloud,
        rgb_image: np.ndarray,
        frame_id: int,
        intrinsic_matrix: np.ndarray,
        camera_matrix_real: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        try:
            corners, tag, (n, score), vis_bgr = self.detect_checkerboard(rgb_image)
            if corners is None or n == 0:
                raise CalibrationError(ErrorCode.NO_MARKER_FOUND, "3", "Checkerboard not found.")

            corners_2d = corners.reshape(-1, 2)
            self.corners_2d_list.append(corners_2d)
            self.frame_ids_list.append(frame_id)

            mean_corner_spacing = self.compute_2d_corner_spacing(corners_2d)
            self.last_mean_corner_spacing = mean_corner_spacing
            self.logger.log_info(f"2D corner mean spacing: {mean_corner_spacing:.2f} px")

            object_points = self._build_object_points()

            if camera_matrix_real is not None:
                self.logger.log_info("Using PnP-based pose estimation (real intrinsics)")
                R_t2c, t_t2c, reproj_err = marker_3d_pose_pnp(
                    corners_2d, object_points, camera_matrix_real, dist_coeffs
                )
                self.logger.log_info(f"PnP reprojection error: {reproj_err:.4f} px")
                tmp_se3 = get_se3(R_t2c, t_t2c)
                marker_3d_points = find_3d_points_from_2d(intrinsic_matrix, pcd, corners)
                pcd.transform(tmp_se3)
            else:
                self.logger.log_info("Using 3D-3D SVD pose estimation (no real intrinsics, fallback)")
                marker_3d_points = find_3d_points_from_2d(intrinsic_matrix, pcd, corners)
                R_t2c, t_t2c = marker_3d_pose(np.array(marker_3d_points), object_points)
                tmp_se3 = get_se3(R_t2c, t_t2c)
                pcd.transform(tmp_se3)

        except CalibrationError:
            raise
        except Exception as e:
            raise CalibrationError(ErrorCode.ESTIMATOR_POSE_FAILED, "3", f"Pose estimation failed: {e}")

        return tmp_se3, frame_id, vis_bgr, pcd, marker_3d_points

    def estimate_pose(
        self,
        pcd: o3d.geometry.PointCloud,
        rgb_image: np.ndarray,
        frame_id: int,
        intrinsic_matrix: np.ndarray,
        camera_matrix_for_pnp: Optional[np.ndarray] = None,
        force_3d_3d: bool = False,
    ):
        camera_matrix_real = (
            None if force_3d_3d
            else (camera_matrix_for_pnp if camera_matrix_for_pnp is not None else self.real_camera_matrix)
        )
        return self.estimate_pose_pnp(
            pcd, rgb_image, frame_id, intrinsic_matrix,
            camera_matrix_real=camera_matrix_real,
            dist_coeffs=self.real_dist_coeffs,
        )

    def get_pose(self, file_list, id: int, robot_pose_type, input_type: str = "ply"):
        """
        Extract robot and camera poses from a list of files.

        Returns:
            (robot_se3, camera_se3, succ_id, vis_image, image_3d, points)
        """
        try:
            ply_data, robot_se3, rgb_from_file, depth_map = self._load_and_parse_files(
                file_list,
                robot_pose_type.rot_type,
                robot_pose_type.rot_unit,
                robot_pose_type.scale_unit,
                input_type=input_type,
            )
        except Exception as e:
            raise CalibrationError(ErrorCode.FILE_LOAD_FAILED, "3", f"Failed to load files: {e}")

        original_scale_x = self.param.get("cam_scale_x", 1.0)
        original_scale_y = self.param.get("cam_scale_y", 1.0)

        # ── RGB + PLY mode ──
        if input_type == "rgb_depth":
            self.logger.log_info("RGB+PLY mode: real RGB corners + PLY 3D lookup → SVD")
            if self.real_camera_matrix is None:
                raise CalibrationError(
                    ErrorCode.INVALID_PARAM, "3",
                    "rgb_depth mode requires camera_matrix in param.",
                )
            try:
                corners, tag, (n, score), vis_bgr = self.detect_checkerboard(rgb_from_file)
                if corners is None or n == 0:
                    raise CalibrationError(ErrorCode.NO_MARKER_FOUND, "3", "Checkerboard not found.")

                corners_2d = corners.reshape(-1, 2)
                self.corners_2d_list.append(corners_2d)
                self.frame_ids_list.append(id)

                marker_3d_points = find_3d_points_from_2d(
                    self.real_camera_matrix, ply_data, corners
                )
                marker_3d_points = np.array(marker_3d_points).reshape(-1, 3)
                self.logger.log_info(f"3D points from PLY: {len(marker_3d_points)}/{len(corners_2d)}")

                object_points = self._build_object_points()
                R_t2c, t_t2c = marker_3d_pose(marker_3d_points, object_points)
                camera_se3 = get_se3(R_t2c, t_t2c)
                ply_data.transform(camera_se3)

            except CalibrationError:
                raise
            except Exception as e:
                raise CalibrationError(ErrorCode.ESTIMATOR_POSE_FAILED, "3", f"rgb_depth failed: {e}")

            self.logger.log_info(f"Robot_se3:\n{robot_se3}")
            self.logger.log_info(f"Camera_se3:\n{camera_se3}")
            return robot_se3, camera_se3, id, vis_bgr, ply_data, marker_3d_points

        # ── RGB mode ──
        if input_type == "rgb":
            self.logger.log_info("RGB mode: using PnP with intrinsics")
            if self.real_camera_matrix is None:
                raise CalibrationError(
                    ErrorCode.INVALID_PARAM, "3",
                    "RGB mode requires camera_matrix in param.",
                )
            try:
                corners, tag, (n, score), vis_bgr = self.detect_checkerboard(rgb_from_file)
                if corners is None or n == 0:
                    raise CalibrationError(ErrorCode.NO_MARKER_FOUND, "3", "Checkerboard not found.")

                corners_2d = corners.reshape(-1, 2)
                self.corners_2d_list.append(corners_2d)
                self.frame_ids_list.append(id)

                object_points = self._build_object_points()
                R_t2c, t_t2c, reproj_err = marker_3d_pose_pnp(
                    corners_2d, object_points,
                    self.real_camera_matrix, self.real_dist_coeffs,
                )
                self.logger.log_info(f"PnP reprojection error: {reproj_err:.4f} px")
                camera_se3 = get_se3(R_t2c, t_t2c)

            except CalibrationError:
                raise
            except Exception as e:
                raise CalibrationError(ErrorCode.ESTIMATOR_POSE_FAILED, "3", f"RGB mode failed: {e}")

            self.logger.log_info(f"Robot_se3:\n{robot_se3}")
            self.logger.log_info(f"Camera_se3:\n{camera_se3}")
            return robot_se3, camera_se3, id, vis_bgr, None, None

        # ── PLY mode (default) ──
        if ply_data is None:
            raise CalibrationError(ErrorCode.FILE_LOAD_FAILED, "3", "PLY data is None")

        max_iterations = 20
        max_spacing_threshold = 40.0
        rgb = corners = None

        for iteration in range(max_iterations):
            try:
                rgb = self._point_cloud_to_rgb_image(ply_data)
            except Exception as e:
                raise CalibrationError(ErrorCode.UNKNOWN, "3", f"PCD→RGB failed: {e}")

            try:
                corners, tag, (n, score), vis_bgr = self.detect_checkerboard(rgb)

                if corners is None or n == 0:
                    if iteration == 0:
                        scale_factor = (
                            max(max_spacing_threshold / self.last_mean_corner_spacing, 0.7)
                            if self.last_mean_corner_spacing is not None
                            and self.last_mean_corner_spacing > max_spacing_threshold
                            else 0.9
                        )
                        if iteration < max_iterations - 1:
                            self._adjust_scale(scale_factor, "Checkerboard not found")
                    else:
                        raise CalibrationError(
                            ErrorCode.NO_MARKER_FOUND, "3",
                            "Checkerboard not found after rescaling."
                        )
                else:
                    mean_spacing = self.compute_2d_corner_spacing(corners.reshape(-1, 2))
                    if mean_spacing > max_spacing_threshold and iteration < max_iterations - 1:
                        self._adjust_scale(
                            0.9, f"Mean spacing ({mean_spacing:.2f}) > {max_spacing_threshold}"
                        )
                    else:
                        self.logger.log_info(f"Mean spacing ({mean_spacing:.2f}) OK, proceeding")
                        break

            except CalibrationError:
                raise
            except Exception as e:
                if iteration == 0:
                    raise CalibrationError(
                        ErrorCode.ESTIMATOR_POSE_FAILED, "3", f"Pose estimation failed: {e}"
                    )
                self.logger.log_warn(f"Error at iteration {iteration}: {e}")
                break

        if corners is None:
            raise CalibrationError(
                ErrorCode.NO_MARKER_FOUND, "3", "Checkerboard not found after rescaling."
            )

        try:
            camera_se3, succ_id, image_2d, image_3d, points = self.estimate_pose(
                ply_data, rgb, id, self.intrinsic_matrix, force_3d_3d=True
            )
        except Exception as e:
            raise CalibrationError(ErrorCode.ESTIMATOR_POSE_FAILED, "3", f"Pose estimation failed: {e}")
        finally:
            self.set_scale_matrix(original_scale_x, original_scale_y)
            self.param["cam_scale_x"] = original_scale_x
            self.param["cam_scale_y"] = original_scale_y

        self.logger.log_info(f"Robot_se3:\n{robot_se3}")
        self.logger.log_info(f"Camera_se3:\n{camera_se3}")
        return robot_se3, camera_se3, succ_id, image_2d, image_3d, points


# ═════════════════════════════════════════════════════════════════════════════
#  Calibration helpers  (crp_calibration.core.calibration — internal helpers)
# ═════════════════════════════════════════════════════════════════════════════

def _se3_to_params(T: np.ndarray) -> np.ndarray:
    rvec = Rot.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([rvec, T[:3, 3]])


def _params_to_se3(p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:6]
    return T


def _so3_log(Rm: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(Rm).as_rotvec()


# ═════════════════════════════════════════════════════════════════════════════
#  Calibration Base  (crp_calibration.core.calibration.Calibration)
# ═════════════════════════════════════════════════════════════════════════════

class Calibration:
    def __init__(self, logger: Logger):
        self.rst: dict = {}
        self.logger = logger

    def execute(self, frames: Frames):
        raise NotImplementedError

    # ── AX = YB (Shah / Li) ──

    def _solve_AX_YB_single(self, A_matrix, B_matrix, method):
        R_b2f = [B[:3, :3] for B in B_matrix]
        t_b2f = [B[:3, 3] for B in B_matrix]
        R_w2c = [A[:3, :3] for A in A_matrix]
        t_w2c = [A[:3, 3] for A in A_matrix]

        R_b2w, t_b2w, R_f2c, t_f2c = cv2.calibrateRobotWorldHandEye(
            R_world2cam=R_w2c,
            t_world2cam=t_w2c,
            R_base2gripper=R_b2f,
            t_base2gripper=t_b2f,
            method=method,
        )
        X = get_se3(R_f2c, t_f2c)
        Y = get_se3(R_b2w, t_b2w)
        return X, Y

    def _solve_AX_YB(self, A_matrix, B_matrix):
        methods = {
            "SHAH": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
            "LI": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
        }
        best_X = best_Y = None
        best_score = float("inf")
        best_name = ""

        for name, method in methods.items():
            try:
                X, Y = self._solve_AX_YB_single(A_matrix, B_matrix, method)
                pos_err, rot_err = self._calc_score(
                    X_matrix=X, Y_matrix=Y, A_matrix=A_matrix, B_matrix=B_matrix
                )
                combined = pos_err + rot_err * 0.1
                self.logger.log_info(
                    f"  [{name}] pos_err={pos_err:.4f}, rot_err={rot_err:.4f}°, combined={combined:.4f}"
                )
                if combined < best_score:
                    best_score = combined
                    best_X, best_Y = X, Y
                    best_name = name
            except Exception as e:
                self.logger.log_info(f"  [{name}] failed: {e}")

        if best_X is None:
            raise RuntimeError("All AX=YB methods failed.")
        self.logger.log_info(f"  → Best AX=YB method: {best_name} (score={best_score:.4f})")
        return best_X, best_Y

    # ── Nonlinear Refinement (LM) ──

    def _refine_nonlinear(self, X_init, Y_init, A_matrix, B_matrix, max_iter=200):
        x0 = np.concatenate([_se3_to_params(X_init), _se3_to_params(Y_init)])

        def residuals(params):
            X = _params_to_se3(params[:6])
            Y = _params_to_se3(params[6:12])
            X_inv = inv_se3(X)
            Y_inv = inv_se3(Y)
            errs = []
            for A, B in zip(A_matrix, B_matrix):
                left = X_inv @ A
                right = B @ Y_inv
                errs.extend(left[:3, 3] - right[:3, 3])
                errs.extend(_so3_log(left[:3, :3] @ right[:3, :3].T) * 10.0)
            return np.array(errs)

        try:
            result = least_squares(
                residuals, x0, method="lm", max_nfev=max_iter, ftol=1e-12, xtol=1e-12
            )
            X_ref = _params_to_se3(result.x[:6])
            Y_ref = _params_to_se3(result.x[6:12])

            ei_p, ei_r = self._calc_score(X_init, Y_init, A_matrix, B_matrix)
            er_p, er_r = self._calc_score(X_ref, Y_ref, A_matrix, B_matrix)
            self.logger.log_info(
                f"  LM refinement: pos {ei_p:.4f}→{er_p:.4f}, rot {ei_r:.4f}→{er_r:.4f}°"
            )
            if (er_p + er_r * 0.1) <= (ei_p + ei_r * 0.1):
                return X_ref, Y_ref
            else:
                self.logger.log_info("  LM refinement did not improve, keeping initial.")
                return X_init, Y_init
        except Exception as e:
            self.logger.log_info(f"  LM refinement failed: {e}, keeping initial.")
            return X_init, Y_init

    # ── Outlier Rejection (LOO) ──

    def _reject_outliers(self, A_matrix, B_matrix, threshold_sigma=2.0):
        n = len(A_matrix)
        if n <= 4:
            return A_matrix, B_matrix, list(range(n))

        loo_errors = []
        for i in range(n):
            A_sub = A_matrix[:i] + A_matrix[i + 1:]
            B_sub = B_matrix[:i] + B_matrix[i + 1:]
            try:
                X_sub, Y_sub = self._solve_AX_YB_single(
                    A_sub, B_sub, cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
                )
                X_inv = inv_se3(X_sub)
                Y_inv = inv_se3(Y_sub)
                left = X_inv @ A_matrix[i]
                right = B_matrix[i] @ Y_inv
                loo_errors.append(np.linalg.norm(left[:3, 3] - right[:3, 3]))
            except Exception:
                loo_errors.append(float("inf"))

        loo_errors = np.array(loo_errors)
        finite_mask = np.isfinite(loo_errors)

        if finite_mask.sum() < 4:
            self.logger.log_info("  Outlier rejection: too few valid LOO results, skipping.")
            return A_matrix, B_matrix, list(range(n))

        median_err = np.median(loo_errors[finite_mask])
        mad = np.median(np.abs(loo_errors[finite_mask] - median_err))
        sigma_est = max(1.4826 * mad, 1e-9)
        inlier_mask = (np.abs(loo_errors - median_err) < threshold_sigma * sigma_est) & finite_mask

        if inlier_mask.sum() < 4:
            sorted_idx = np.argsort(loo_errors)
            inlier_mask = np.zeros(n, dtype=bool)
            inlier_mask[sorted_idx[: max(4, n // 2)]] = True

        valid_indices = np.where(inlier_mask)[0].tolist()
        n_rejected = n - len(valid_indices)
        if n_rejected > 0:
            rejected = np.where(~inlier_mask)[0].tolist()
            self.logger.log_info(f"  Outlier rejection: removed {n_rejected} frame(s): {rejected}")
            for ri in rejected:
                self.logger.log_info(f"    frame {ri}: LOO error = {loo_errors[ri]:.4f}")

        A_filtered = [A_matrix[i] for i in valid_indices]
        B_filtered = [B_matrix[i] for i in valid_indices]
        return A_filtered, B_filtered, valid_indices

    # ── Score ──

    def _calc_score(self, X_matrix, Y_matrix, A_matrix, B_matrix):
        X_inv = inv_se3(X_matrix)
        Y_inv = inv_se3(Y_matrix)
        pos_errs, rot_errs = [], []

        for A, B in zip(A_matrix, B_matrix):
            left = X_inv @ A
            right = B @ Y_inv
            pos_errs.append(np.linalg.norm(left[:3, 3] - right[:3, 3]))
            R_err = left[:3, :3] @ np.linalg.inv(right[:3, :3])
            trace_val = np.clip(np.trace(R_err), -1, 3)
            rot_errs.append(np.degrees(np.arccos((trace_val - 1) / 2)))

        avg_pos = float(np.mean(pos_errs))
        avg_rot = float(np.mean(rot_errs))
        self.logger.log_info(f"  Avg Position Error: {avg_pos:.4f}")
        self.logger.log_info(f"  Avg Rotation Error: {avg_rot:.4f}°")
        return avg_pos, avg_rot


# ═════════════════════════════════════════════════════════════════════════════
#  HandEyeCalibration  (crp_calibration.core.calibration.HandEyeCalibration)
# ═════════════════════════════════════════════════════════════════════════════

class HandEyeCalibration(Calibration):
    """Eye-in-hand calibration via AX = YB.

    A = inv(camera_pose) = cam2world,  B = base2flange
    X = flange2cam,  Y = base2world
    """

    def __init__(self, logger: Logger, param: dict):
        super().__init__(logger)
        self.param = param

    def execute(self, frames: Frames) -> dict:
        A_matrix, B_matrix, valid_ids = [], [], []

        for fid, frame in frames.frames.items():
            if frame.camera_pose is None or frame.robot_pose is None:
                continue
            if np.allclose(frame.camera_pose, np.eye(4)) or np.allclose(frame.robot_pose, np.eye(4)):
                continue
            valid_ids.append(fid)
            A_matrix.append(inv_se3(frame.camera_pose))  # cam2world (eye-in-hand)
            B_matrix.append(frame.robot_pose)

        if len(A_matrix) < 3:
            raise ValueError(f"Insufficient valid frames: {len(A_matrix)} < 3")

        self.logger.log_info(f"HandEye calibration with {len(A_matrix)} valid frames")
        self.logger.log_info(
            "AX = YB | A: cam2world, X: flange2cam, Y: base2world, B: base2flange"
        )

        # Step 1: Outlier rejection
        self.logger.log_info("Step 1: Outlier rejection (leave-one-out)")
        A_clean, B_clean, inlier_idx = self._reject_outliers(A_matrix, B_matrix)
        clean_valid_ids = [valid_ids[i] for i in inlier_idx]
        self.logger.log_info(
            f"  Using {len(A_clean)}/{len(A_matrix)} frames after outlier rejection"
        )

        # Step 2: Solve AX=YB
        self.logger.log_info("Step 2: AX=YB solver comparison (Shah vs Li)")
        try:
            X_matrix, Y_matrix = self._solve_AX_YB(A_clean, B_clean)
        except Exception as e:
            raise RuntimeError(f"AX=YB solution failed: {e}") from e

        # Step 3: LM refinement
        self.logger.log_info("Step 3: Nonlinear refinement (LM)")
        X_matrix, Y_matrix = self._refine_nonlinear(X_matrix, Y_matrix, A_clean, B_clean)

        # Step 4: Final score
        self.logger.log_info("Step 4: Final score")
        score, rot_score = self._calc_score(
            A_matrix=A_clean, X_matrix=X_matrix, Y_matrix=Y_matrix, B_matrix=B_clean
        )

        # Per-frame errors
        X_inv = inv_se3(X_matrix)
        Y_inv = inv_se3(Y_matrix)
        fid_to_row = {fid: i for i, fid in enumerate(clean_valid_ids)}
        frame_results = []
        for fid in frames.frames.keys():
            idx = fid_to_row.get(fid)
            if idx is not None:
                left = X_inv @ A_clean[idx]
                right = B_clean[idx] @ Y_inv
                frame_results.append(float(np.linalg.norm(left[:3, 3] - right[:3, 3])))
            else:
                frame_results.append(-1.0)

        self.rst = {
            "A": A_clean,
            "X": X_matrix,
            "Y": Y_matrix,
            "B": B_clean,
            "flange2cam": X_matrix,
            "base2world": Y_matrix,
            "cam2flange": inv_se3(X_matrix),
            "world2base": inv_se3(Y_matrix),
            "score": float(score),
            "rotation_score": float(rot_score),
            "frame_results": frame_results,
            "valid_ids": clean_valid_ids,
            "n_outliers_rejected": len(A_matrix) - len(A_clean),
        }
        return self.rst


# ═════════════════════════════════════════════════════════════════════════════
#  FixedEyeCalibration  (crp_calibration.core.calibration.FixedEyeCalibration)
# ═════════════════════════════════════════════════════════════════════════════

class FixedEyeCalibration(Calibration):
    def __init__(self, logger: Logger, param: dict):
        super().__init__(logger)
        self.param = param

    def execute(self, frames: Frames) -> dict:
        A_matrix, B_matrix, valid_ids = [], [], []

        for fid, frame in frames.frames.items():
            if frame.camera_pose is None or frame.robot_pose is None:
                continue
            if np.allclose(frame.camera_pose, np.eye(4)) or np.allclose(frame.robot_pose, np.eye(4)):
                continue
            valid_ids.append(fid)
            A_matrix.append(frame.camera_pose)
            B_matrix.append(frame.robot_pose)

        if len(A_matrix) < 3:
            raise ValueError(f"Insufficient valid frames: {len(A_matrix)} < 3")

        self.logger.log_info(f"FixedEye calibration with {len(A_matrix)} valid frames")
        self.logger.log_info(
            "AX = YB | A: world2cam, X: flange2cam, Y: base2world, B: base2flange"
        )

        # Step 1: Outlier rejection
        self.logger.log_info("Step 1: Outlier rejection (leave-one-out)")
        A_clean, B_clean, inlier_idx = self._reject_outliers(A_matrix, B_matrix)
        clean_valid_ids = [valid_ids[i] for i in inlier_idx]
        self.logger.log_info(
            f"  Using {len(A_clean)}/{len(A_matrix)} frames after outlier rejection"
        )

        # Step 2: Solve AX=YB
        self.logger.log_info("Step 2: AX=YB solver comparison (Shah vs Li)")
        try:
            X_matrix, Y_matrix = self._solve_AX_YB(A_clean, B_clean)
        except Exception as e:
            raise RuntimeError(f"AX=YB solution failed: {e}")

        # Step 3: LM refinement
        self.logger.log_info("Step 3: Nonlinear refinement (LM)")
        X_matrix, Y_matrix = self._refine_nonlinear(X_matrix, Y_matrix, A_clean, B_clean)

        # Step 4: Final score
        self.logger.log_info("Step 4: Final score")
        score, rot_score = self._calc_score(
            A_matrix=A_clean, X_matrix=X_matrix, Y_matrix=Y_matrix, B_matrix=B_clean
        )

        # Per-frame errors
        X_inv = inv_se3(X_matrix)
        Y_inv = inv_se3(Y_matrix)
        fid_to_row = {fid: i for i, fid in enumerate(clean_valid_ids)}
        frame_results = []
        for fid in frames.frames.keys():
            idx = fid_to_row.get(fid)
            if idx is not None:
                left = X_inv @ A_clean[idx]
                right = B_clean[idx] @ Y_inv
                frame_results.append(float(np.linalg.norm(left[:3, 3] - right[:3, 3])))
            else:
                frame_results.append(-1.0)

        self.rst = {
            "A": A_clean,
            "X": X_matrix,
            "Y": Y_matrix,
            "B": B_clean,
            "base2cam": X_matrix,
            "world2flange": Y_matrix,
            "cam2base": inv_se3(X_matrix),
            "flange2world": inv_se3(Y_matrix),
            "score": float(score),
            "rotation_score": float(rot_score),
            "frame_results": frame_results,
            "valid_ids": clean_valid_ids,
            "n_outliers_rejected": len(A_matrix) - len(A_clean),
        }
        return self.rst


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Logger
    "Logger",
    # Transform
    "inv_se3",
    "get_se3",
    "inverse_transform",
    "rotation_to_matrix",
    "euler_to_matrix",
    "quaternion_to_matrix",
    "rotation_vector_to_matrix",
    # IO
    "read_ply_file",
    "save_image_file",
    "save_ply_file",
    "read_json_file",
    "set_ply_prefix",
    "set_debug_path",
    # Error
    "ErrorCode",
    "ErrorSeverity",
    "CalibrationError",
    # Frames
    "Frame",
    "Frames",
    # Estimator
    "marker_3d_pose",
    "marker_3d_pose_pnp",
    "find_3d_points_from_2d",
    "CameraEstimator",
    # Calibration
    "FixedEyeCalibration",
    "HandEyeCalibration",
]
