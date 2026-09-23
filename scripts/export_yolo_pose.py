"""Dump Waymo slice to YOLO-pose folders (images/ + labels/ + data.yaml)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from pose_av.data_waymo import load_slice, resolve_waymo_root
    from pose_av.export_yolo import export_yolo_pose

    root = resolve_waymo_root(str(cfg.data.waymo_root))
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    max_frames = cfg.data.get("max_frames")
    slice_data = load_slice(
        root,
        segment=str(cfg.data.segment),
        split=str(cfg.data.split),
        max_frames=None if max_frames in (None, "null") else int(max_frames),
    )
    out = ROOT / str(cfg.export_dir)
    stats = export_yolo_pose(slice_data, out, val_frac=float(cfg.val_frac))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
