"""Fine-tune Ultralytics YOLO-pose on the exported Waymo slice."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _device(cfg_device: str):
    if cfg_device != "auto":
        return cfg_device
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from ultralytics import YOLO

    from pose_av.runlog import log_remote, write_run
    from pose_av.seed import set_seed

    set_seed(int(cfg.seed))
    yaml_path = ROOT / str(cfg.export_dir) / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Run export first: missing {yaml_path}")

    model = YOLO(str(cfg.model.weights))
    device = _device(str(cfg.train.device))
    amp = bool(cfg.train.amp)
    tcfg = OmegaConf.to_container(cfg.train, resolve=True)
    assert isinstance(tcfg, dict)
    reserved = {
        "epochs",
        "imgsz",
        "batch",
        "workers",
        "device",
        "amp",
        "patience",
        "project",
        "name",
    }
    extra = {k: v for k, v in tcfg.items() if k not in reserved and v is not None}

    results = model.train(
        data=str(yaml_path),
        epochs=int(cfg.train.epochs),
        imgsz=int(cfg.train.imgsz),
        batch=int(cfg.train.batch),
        workers=int(cfg.train.workers),
        device=device,
        amp=amp,
        patience=int(cfg.train.patience),
        project=str(ROOT / cfg.train.project),
        name=str(cfg.train.name),
        exist_ok=True,
        pretrained=True,
        seed=int(cfg.seed),
        **extra,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    metrics = {
        "best_weights": str(best),
        "save_dir": str(save_dir),
        "epochs": int(cfg.train.epochs),
        "imgsz": int(cfg.train.imgsz),
        "batch": int(cfg.train.batch),
        "amp": amp,
        "workers": int(cfg.train.workers),
        "device": str(device),
        "model": str(cfg.model.weights),
    }
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    hydra_out = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    path = write_run(hydra_out, metrics, cfg_resolved)
    log_remote(metrics, cfg_resolved, dict(cfg_resolved.get("logging") or {}), hydra_out)
    print(json.dumps(metrics, indent=2))
    print("wrote", path)
    print("eval with: python scripts/eval_pose.py model.weights=" + str(best))


if __name__ == "__main__":
    main()
