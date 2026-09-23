"""Hydra CLI: measure YOLO-pose on a Waymo parquet slice (GP2 protocol).

Example:
    python scripts/eval_pose.py
    python scripts/eval_pose.py model=yolov8n_pose data.max_frames=5 protocol=full_frame
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _device(cfg_device: str) -> str:
    if cfg_device != "auto":
        return cfg_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    from pose_av.data_waymo import load_slice, resolve_waymo_root
    from pose_av.evaluate import run_protocols
    from pose_av.runlog import log_remote, write_run
    from pose_av.seed import set_seed
    from pose_av.yolo_backend import YoloPoseBackend

    set_seed(int(cfg.seed))
    device = _device(str(cfg.device))
    root = resolve_waymo_root(str(cfg.data.waymo_root))
    if not root.is_absolute():
        root = (ROOT / root).resolve()

    mf = cfg.data.max_frames
    max_frames = None if mf in (None, "null", "None") else int(mf)
    slice_data = load_slice(
        root,
        segment=str(cfg.data.segment),
        split=str(cfg.data.split),
        max_frames=max_frames,
        subset=str(cfg.data.get("subset", "all")),
        val_frac=float(cfg.data.get("val_frac", 0.2)),
    )
    backend = YoloPoseBackend(
        weights=str(cfg.model.weights),
        device=device,
        imgsz=int(cfg.model.imgsz),
        conf=float(cfg.model.conf),
        crop_imgsz=int(cfg.model.crop_imgsz),
        crop_conf=float(cfg.model.crop_conf),
        kpt_conf=float(cfg.model.kpt_conf),
        crop_kpt_conf=float(cfg.model.crop_kpt_conf),
        crop_pad=float(cfg.model.crop_pad),
    )
    protocol = str(cfg.protocol)
    metrics = run_protocols(
        slice_data,
        backend,
        iou_match=float(cfg.iou_match),
        do_full_frame=protocol in ("both", "full_frame"),
        do_gt_crop=protocol in ("both", "gt_crop"),
        progress_every=int(cfg.progress_every),
    )
    metrics.update(
        {
            "model": str(cfg.model.weights),
            "device": device,
            "imgsz": int(cfg.model.imgsz),
            "conf": float(cfg.model.conf),
            "segment": str(cfg.data.segment),
            "paths": slice_data.paths,
        }
    )

    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    hydra_out = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metrics_path = write_run(hydra_out, metrics, cfg_resolved)
    print(json.dumps(metrics, indent=2))
    print("wrote", metrics_path)
    try:
        remote = log_remote(metrics, cfg_resolved, dict(cfg_resolved.get("logging") or {}), hydra_out)
    except Exception as exc:
        print("remote logging failed:", type(exc).__name__, exc)
        remote = {}
    if any(remote.values()):
        metrics["remote"] = remote
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
