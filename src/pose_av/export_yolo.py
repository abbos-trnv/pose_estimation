"""Export a Waymo parquet slice to Ultralytics YOLO-pose layout."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from pose_av.boxes import center_wh_to_xyxy
from pose_av.data_waymo import SliceTables, gt_keypoints
from pose_av.evaluate import decode_jpeg
from pose_av.metrics import waymo14_to_coco17

# COCO person flip indices for kpt_shape [17, 3]
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _split_keys(keys: list, val_frac: float) -> tuple[list, list]:
    """Hold out the last val_frac of time-sorted frames (less leakage than a shuffle)."""
    if not keys:
        return [], []
    n_val = max(1, int(round(len(keys) * val_frac)))
    n_val = min(n_val, len(keys) - 1) if len(keys) > 1 else 0
    train, val = keys[:-n_val] if n_val else keys, keys[-n_val:] if n_val else []
    return train, val


def _yolo_line(box: dict, coco_xy: np.ndarray, coco_vis: np.ndarray, w_img: int, h_img: int) -> str | None:
    x1, y1, x2, y2 = center_wh_to_xyxy(box["cx"], box["cy"], box["w"], box["h"])
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w_img), x2), min(float(h_img), y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    cx = ((x1 + x2) / 2.0) / w_img
    cy = ((y1 + y2) / 2.0) / h_img
    nw, nh = bw / w_img, bh / h_img
    parts = [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]
    vis_any = False
    for i in range(17):
        x = float(np.clip(coco_xy[i, 0] / w_img, 0.0, 1.0))
        y = float(np.clip(coco_xy[i, 1] / h_img, 0.0, 1.0))
        v = 2 if coco_vis[i] > 0 else 0
        vis_any = vis_any or v > 0
        parts.append(f"{x:.6f} {y:.6f} {int(v)}")
    if not vis_any:
        return None
    return " ".join(parts)


def export_yolo_pose(
    slice_data: SliceTables,
    out_dir: Path,
    *,
    val_frac: float = 0.2,
) -> dict:
    out_dir = Path(out_dir)
    train_keys, val_keys = _split_keys(slice_data.keys, val_frac)
    stats = {"train_images": 0, "val_images": 0, "train_instances": 0, "val_instances": 0}

    for split, keys in (("train", train_keys), ("val", val_keys)):
        img_dir = out_dir / "images" / split
        lab_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for key in keys:
            im = decode_jpeg(slice_data.images[key])
            if im is None:
                continue
            h, w = im.shape[:2]
            box_by_oid = {b["oid"]: b for b in slice_data.boxes.get(key, [])}
            lines: list[str] = []
            for rec in slice_data.hkps.get(key, []):
                b = box_by_oid.get(rec["oid"])
                if b is None:
                    continue
                gt_xy, gt_vis = gt_keypoints(rec)
                if gt_vis.sum() == 0:
                    continue
                coco_xy, coco_vis = waymo14_to_coco17(gt_xy, gt_vis)
                line = _yolo_line(b, coco_xy, coco_vis, w, h)
                if line:
                    lines.append(line)
            if not lines:
                continue
            stem = f"{key[0]}_{key[1]}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), im)
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats[f"{split}_images"] += 1
            stats[f"{split}_instances"] += len(lines)

    yaml_path = out_dir / "data.yaml"
    payload = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [17, 3],
        "flip_idx": FLIP_IDX,
        "names": {0: "person"},
    }
    yaml_path.write_text(yaml.dump(payload, sort_keys=False), encoding="utf-8")
    stats["yaml"] = str(yaml_path)
    return stats
