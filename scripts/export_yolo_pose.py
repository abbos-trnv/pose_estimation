"""Dump Waymo slice(s) to YOLO-pose folders.

Default: one segment, time split.
With configs/data/gp4_segments.yaml + downloaded images: scene split
(train = extra clips, val = original Kaggle segment).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _try_load(root: Path, segment: str, split: str):
    from pose_av.data_waymo import find_parquet, load_slice

    try:
        find_parquet(root, "camera_image", segment)
        find_parquet(root, "camera_box", segment)
    except FileNotFoundError:
        print("skip (no images yet)", segment)
        return None
    return load_slice(root, segment, split=split, max_frames=None, subset="all")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from pose_av.data_waymo import load_slice, resolve_waymo_root
    from pose_av.export_yolo import export_frames, export_yolo_pose, write_data_yaml

    root = resolve_waymo_root(str(cfg.data.waymo_root))
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    out = ROOT / str(cfg.export_dir)
    seg_file = ROOT / "configs" / "data" / "gp4_segments.yaml"
    if seg_file.exists():
        spec = OmegaConf.load(seg_file)
        train_ids = list(spec.get("train_segments") or [])
        val_id = spec.get("val_segment")
        have_train = False
        stats = {"train_images": 0, "val_images": 0, "train_instances": 0, "val_instances": 0, "skipped": []}
        for sid in train_ids:
            sl = _try_load(root, str(sid), str(cfg.data.split))
            if sl is None:
                stats["skipped"].append(str(sid))
                continue
            have_train = True
            export_frames(sl, out, "train", stats, prefix=str(sid)[:12])
        if val_id:
            sl = _try_load(root, str(val_id), str(cfg.data.split))
            if sl is not None:
                export_frames(sl, out, "val", stats, prefix="val")
        if have_train and stats["val_images"]:
            stats["yaml"] = str(write_data_yaml(out))
            print(json.dumps(stats, indent=2))
            return
        print("scene split incomplete, fallback to single-segment time split")

    max_frames = cfg.data.get("max_frames")
    slice_data = load_slice(
        root,
        segment=str(cfg.data.segment),
        split=str(cfg.data.split),
        max_frames=None if max_frames in (None, "null") else int(max_frames),
    )
    stats = export_yolo_pose(slice_data, out, val_frac=float(cfg.val_frac))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
