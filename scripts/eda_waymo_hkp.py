"""EDA over local Waymo v2 training/camera_hkp parquet files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HKP_DIR = ROOT / "data" / "waymo_v2" / "training" / "camera_hkp" / "camera_hkp"

COL_TYPE = "[CameraHumanKeypointsComponent].camera_keypoints[*].type"
COL_X = "[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.location_px.x"
COL_Y = "[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.location_px.y"
COL_OCC = "[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.visibility.is_occluded"
COL_X3 = "[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_3d.location_m.x"

TYPE_NAME = {
    1: "nose",
    5: "left_shoulder",
    13: "right_shoulder",
    6: "left_elbow",
    14: "right_elbow",
    7: "left_wrist",
    15: "right_wrist",
    8: "left_hip",
    16: "right_hip",
    9: "left_knee",
    17: "right_knee",
    10: "left_ankle",
    18: "right_ankle",
    19: "forehead",
    20: "head_center",
}


def main() -> None:
    files = sorted(p for p in HKP_DIR.glob("*.parquet"))
    n_files = len(files)
    n_empty = 0
    n_objects = 0
    n_with_2d = 0
    n_with_3d = 0
    kpt_counts: list[int] = []
    type_n = Counter()
    type_occ = Counter()
    cam_n = Counter()
    xs: list[float] = []
    ys: list[float] = []
    frames = set()
    segments_nonempty = 0

    for path in files:
        table = pq.read_table(path, columns=[
            "key.segment_context_name",
            "key.frame_timestamp_micros",
            "key.camera_name",
            "key.camera_object_id",
            COL_TYPE,
            COL_X,
            COL_Y,
            COL_OCC,
            COL_X3,
        ])
        n = table.num_rows
        if n == 0:
            n_empty += 1
            continue
        segments_nonempty += 1
        n_objects += n
        seg = table.column("key.segment_context_name").to_pylist()
        ts = table.column("key.frame_timestamp_micros").to_pylist()
        cam = table.column("key.camera_name").to_pylist()
        types = table.column(COL_TYPE).to_pylist()
        xs_col = table.column(COL_X).to_pylist()
        ys_col = table.column(COL_Y).to_pylist()
        occ = table.column(COL_OCC).to_pylist()
        x3 = table.column(COL_X3).to_pylist()

        for i in range(n):
            frames.add((seg[i], ts[i], cam[i]))
            cam_n[int(cam[i])] += 1
            tlist = types[i] or []
            kpt_counts.append(len(tlist))
            if xs_col[i]:
                n_with_2d += 1
                xs.extend(xs_col[i])
                ys.extend(ys_col[i] or [])
            if x3[i]:
                n_with_3d += 1
            olist = occ[i] or []
            for j, t in enumerate(tlist):
                t = int(t)
                type_n[t] += 1
                if j < len(olist) and olist[j]:
                    type_occ[t] += 1

    k = np.asarray(kpt_counts, dtype=np.float64) if kpt_counts else np.array([0.0])
    by_type = []
    for t, name in TYPE_NAME.items():
        tot = type_n[t]
        occ_n = type_occ[t]
        by_type.append({
            "type": t,
            "name": name,
            "count": tot,
            "occluded": occ_n,
            "occluded_pct": round(100.0 * occ_n / tot, 2) if tot else None,
        })

    summary = {
        "parquet_files": n_files,
        "empty_segments": n_empty,
        "nonempty_segments": segments_nonempty,
        "objects": n_objects,
        "unique_camera_frames": len(frames),
        "objects_with_2d": n_with_2d,
        "objects_with_3d_in_camera_component": n_with_3d,
        "keypoints_per_object": {
            "mean": float(k.mean()) if len(k) else 0,
            "median": float(np.median(k)) if len(k) else 0,
            "min": int(k.min()) if len(k) else 0,
            "max": int(k.max()) if len(k) else 0,
        },
        "xy_px": {
            "x_min": float(np.min(xs)) if xs else None,
            "x_max": float(np.max(xs)) if xs else None,
            "y_min": float(np.min(ys)) if ys else None,
            "y_max": float(np.max(ys)) if ys else None,
        },
        "objects_per_camera_id": dict(sorted(cam_n.items())),
        "by_joint": by_type,
    }
    out = ROOT / "reports" / "gp1_eda_stats.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
