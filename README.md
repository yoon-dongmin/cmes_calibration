# Camera Calibration

Standalone calibration scripts for **fixed-eye (eye-to-hand)** and **hand-eye** camera setups.  
All logic lives in a single self-contained module (`calib_module.py`) — no external CRP packages required.

---

## Supported Input Types

| Mode        | File format          | Method              |
|-------------|----------------------|---------------------|
| `rgb`       | `.png / .jpg / .bmp` | PnP (2D corners)    |
| `ply`       | `.ply`               | SVD (3D points)     |
| `rgb_depth` | `.png` + `.ply`      | 2D corners + 3D lookup |

---

## File Naming Convention

Each capture file must encode the robot pose in its name:

```
{index}_{tx}_{ty}_{tz}_{rx}_{ry}_{rz}.ext
```

Example: `000_-586.8_-82.5_617.3_178.89_-0.34_-92.23.png`

---

## Requirements

- Python >= 3.8

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Fixed-Eye Calibration (eye-to-hand)

Camera is fixed; robot moves the target board.

```bash
python fixed_eye_calibration.py \
    --data_dir data/260423 \
    --rows 8 --cols 7 --size 20.0 \
    --input_type rgb \
    --intrinsic 616.18 616.29 326.41 239.17 \
    --rot_type euler --rot_unit deg --pos_unit mm
```

### Hand-Eye Calibration

Camera is mounted on the robot end-effector.

```bash
python hand_eye_calibration.py \
    --data_dir data/260413 \
    --rows 8 --cols 7 --size 20.0 \
    --input_type rgb_depth \
    --intrinsic 616.18 616.29 326.41 239.17 \
    --rot_type euler --rot_unit deg --pos_unit mm
```

---

## Key Arguments

| Argument        | Default              | Description                                      |
|-----------------|----------------------|--------------------------------------------------|
| `--data_dir`    | `data/260423`        | Directory containing input files                 |
| `--output_dir`  | `./calibration_output` | Directory for results                          |
| `--rows`        | `8`                  | Number of inner corner rows on the checkerboard  |
| `--cols`        | `7`                  | Number of inner corner columns                   |
| `--size`        | `20.0`               | Square size in mm                                |
| `--input_type`  | `rgb`                | `rgb` / `ply` / `rgb_depth`                      |
| `--intrinsic`   | *(built-in default)* | Camera intrinsics: `fx fy cx cy`                 |
| `--dist_coeffs` | all zeros            | OpenCV distortion coefficients                   |
| `--rot_type`    | `euler`              | Rotation representation: `euler / rotvec / quat` |
| `--rot_unit`    | `deg`                | `deg` or `rad`                                   |
| `--pos_unit`    | `mm`                 | `mm` or `m`                                      |
| `--euler_conv`  | `xyz`                | Euler angle convention                           |
| `--debug`       | off                  | Save per-image debug visualizations              |
| `--reference`   | *(none)*             | Path to a reference `.npz` for result comparison |

---

## Output

Results are saved to `--output_dir`:

```
calibration_output/
├── calibration_result.yml      # Human-readable result
├── calibration_result.npz      # NumPy archive (for downstream use)
├── projected_corners_2d.jpg    # Reprojection verification image
└── 000_detected.png            # Per-image corner detection (always saved)
```

Debug images (saved only with `--debug`):

```
calibration_output/
└── clahe_image.png             # Preprocessing pipeline visualization
```

---

## Data Directory Structure

```
data/
└── 260423/                     # Session folder (YYMMDD)
    ├── intrinsics.json         # Optional: overrides --intrinsic
    ├── 000_tx_ty_tz_rx_ry_rz.png
    ├── 000_tx_ty_tz_rx_ry_rz.ply   # Required for rgb_depth / ply modes
    └── ...
```

`intrinsics.json` format:

```json
{
  "fx": 616.178,
  "fy": 616.295,
  "cx": 326.410,
  "cy": 239.175,
  "dist_coeffs": [0, 0, 0, 0, 0]
}
```
