"""Load one Waymo Perception parquet slice (image + box + camera_hkp)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from pose_av.keypoints import CAMERA_ORDER

COL_TYPE = "[CameraHumanKeypointsComponent].camera_keypoints[*].type"
COL_X = "[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.location_px.x"
COL_Y = "[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.location_px.y"
COL_OCC = "[CameraHumanKeypointsComponent].camera_keypoints[*].keypoint_2d.visibility.is_occluded"
COL_IMG = "[CameraImageComponent].image"
COL_CX = "[CameraBoxComponent].box.center.x"
COL_CY = "[CameraBoxComponent].box.center.y"
COL_SX = "[CameraBoxComponent].box.size.x"
COL_SY = "[CameraBoxComponent].box.size.y"

FrameKey = tuple[int, int]


@dataclass
class SliceTables:
    images: dict[FrameKey, bytes]
    boxes: dict[FrameKey, list[dict]]
    hkps: dict[FrameKey, list[dict]]
    keys: list[FrameKey]
    paths: dict[str, str]


def find_parquet(root: Path, kind: str, segment: str) -> Path:
    hits = list(root.glob(f"**/{kind}/{segment}.parquet"))
    hits += list(root.glob(f"**/{kind}/**/{segment}.parquet"))
    if not hits:
        raise FileNotFoundError(f"{kind} parquet for segment {segment} under {root}")
    return hits[0]


def resolve_waymo_root(configured: str | Path) -> Path:
    """Local tree, or all Kaggle datasets under /kaggle/input (several zips)."""
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        return kaggle
    return Path(configured)


def load_slice(
    root: Path,
    segment: str,
    split: str = "training",
    max_frames: int | None = 40,
    subset: str = "all",
    val_frac: float = 0.2,
) -> SliceTables:
    img_path = find_parquet(root, "camera_image", segment)
    box_path = find_parquet(root, "camera_box", segment)
    hkp_path = find_parquet(root, "camera_hkp", segment)

    img_t = pq.read_table(
        img_path,
        columns=["key.frame_timestamp_micros", "key.camera_name", COL_IMG],
    )
    box_t = pq.read_table(
        box_path,
        columns=[
            "key.frame_timestamp_micros",
            "key.camera_name",
            "key.camera_object_id",
            COL_CX,
            COL_CY,
            COL_SX,
            COL_SY,
        ],
    )
    hkp_t = pq.read_table(
        hkp_path,
        columns=[
            "key.frame_timestamp_micros",
            "key.camera_name",
            "key.camera_object_id",
            COL_TYPE,
            COL_X,
            COL_Y,
            COL_OCC,
        ],
    )

    images: dict[FrameKey, bytes] = {}
    ts_i = img_t.column("key.frame_timestamp_micros").to_pylist()
    cam_i = img_t.column("key.camera_name").to_pylist()
    blob = img_t.column(COL_IMG).to_pylist()
    for i in range(img_t.num_rows):
        images[(int(ts_i[i]), int(cam_i[i]))] = blob[i]

    boxes: dict[FrameKey, list[dict]] = defaultdict(list)
    ts_b = box_t.column("key.frame_timestamp_micros").to_pylist()
    cam_b = box_t.column("key.camera_name").to_pylist()
    oid_b = box_t.column("key.camera_object_id").to_pylist()
    cx = box_t.column(COL_CX).to_pylist()
    cy = box_t.column(COL_CY).to_pylist()
    sx = box_t.column(COL_SX).to_pylist()
    sy = box_t.column(COL_SY).to_pylist()
    for i in range(box_t.num_rows):
        boxes[(int(ts_b[i]), int(cam_b[i]))].append(
            {"oid": oid_b[i], "cx": cx[i], "cy": cy[i], "w": sx[i], "h": sy[i]}
        )

    hkps: dict[FrameKey, list[dict]] = defaultdict(list)
    ts_h = hkp_t.column("key.frame_timestamp_micros").to_pylist()
    cam_h = hkp_t.column("key.camera_name").to_pylist()
    oid_h = hkp_t.column("key.camera_object_id").to_pylist()
    types = hkp_t.column(COL_TYPE).to_pylist()
    xs = hkp_t.column(COL_X).to_pylist()
    ys = hkp_t.column(COL_Y).to_pylist()
    occ = hkp_t.column(COL_OCC).to_pylist()
    for i in range(hkp_t.num_rows):
        hkps[(int(ts_h[i]), int(cam_h[i]))].append(
            {
                "oid": oid_h[i],
                "types": types[i] or [],
                "x": xs[i] or [],
                "y": ys[i] or [],
                "occ": occ[i] or [],
            }
        )

    keys = [k for k in images if k in hkps]
    keys.sort()
    if max_frames is not None:
        keys = keys[: int(max_frames)]
    keys = apply_frame_subset(keys, subset=subset, val_frac=val_frac)

    return SliceTables(
        images=images,
        boxes=dict(boxes),
        hkps=dict(hkps),
        keys=keys,
        paths={
            "image": str(img_path),
            "box": str(box_path),
            "hkp": str(hkp_path),
            "split": split,
            "subset": subset,
        },
    )


def apply_frame_subset(keys: list, subset: str = "all", val_frac: float = 0.2) -> list:
    """Time-sorted holdout, same rule as YOLO export (last val_frac = val)."""
    if subset in (None, "all"):
        return keys
    if not keys:
        return keys
    n_val = max(1, int(round(len(keys) * val_frac)))
    n_val = min(n_val, len(keys) - 1) if len(keys) > 1 else 0
    if subset == "val":
        return keys[-n_val:] if n_val else keys
    if subset == "train":
        return keys[:-n_val] if n_val else keys
    raise ValueError(f"unknown subset {subset}")


def gt_keypoints(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((14, 2), dtype=np.float64)
    vis = np.zeros((14,), dtype=np.float64)
    tmap = {int(t): j for j, t in enumerate(rec["types"])}
    for k, t in enumerate(CAMERA_ORDER):
        if t not in tmap:
            continue
        j = tmap[t]
        xy[k] = [rec["x"][j], rec["y"][j]]
        vis[k] = 1.0
    return xy, vis
