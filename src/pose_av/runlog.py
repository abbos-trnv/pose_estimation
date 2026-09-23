"""Local run folder + optional cloud loggers (W&B, MLflow).

Local `runs/` is always written (Kaggle filesystem is ephemeral).
Cloud backends are no-ops unless the package is installed and a key/URI is set —
so the same script works on a laptop and on a Kaggle GPU.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def flatten_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    """Nested dict → scalar map for W&B / MLflow. Non-numeric values skipped."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            name = f"{prefix}{key}" if prefix else str(key)
            out.update(flatten_metrics(val, f"{name}/"))
        return out
    if isinstance(obj, bool):
        return {}
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        key = prefix[:-1] if prefix.endswith("/") else prefix
        if key:
            out[key] = float(obj)
    return out


def write_run(run_dir: Path, metrics: dict[str, Any], config: dict[str, Any] | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if config is not None:
        try:
            from omegaconf import OmegaConf

            (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(config), encoding="utf-8")
        except Exception:
            (run_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str), encoding="utf-8"
            )
    return metrics_path


def _wandb_key_present() -> bool:
    return bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY_FILE"))


def log_wandb(
    metrics: dict[str, Any],
    config: dict[str, Any] | None,
    *,
    project: str,
    run_name: str | None,
    entity: str | None,
    run_dir: Path | None,
) -> str | None:
    if not _wandb_key_present():
        print("wandb: skip (set WANDB_API_KEY to log to wandb.ai)")
        return None
    try:
        import wandb
    except ImportError:
        print("wandb: skip (pip install wandb)")
        return None

    run = wandb.init(
        project=project,
        entity=entity or None,
        name=run_name or None,
        config=config or {},
        finish_previous=True,
    )
    run.log(flatten_metrics(metrics))
    if run_dir is not None:
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            run.save(str(metrics_file), policy="now")
        cfg_file = run_dir / "config.yaml"
        if cfg_file.exists():
            run.save(str(cfg_file), policy="now")
    url = run.url
    run.finish()
    print("wandb:", url)
    return url


def log_mlflow(
    metrics: dict[str, Any],
    config: dict[str, Any] | None,
    *,
    tracking_uri: str,
    experiment: str,
    run_name: str | None,
    run_dir: Path | None,
) -> str | None:
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        print("mlflow: skip (set logging.mlflow.tracking_uri or MLFLOW_TRACKING_URI)")
        return None
    try:
        import mlflow
    except ImportError:
        print("mlflow: skip (pip install mlflow)")
        return None

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        flat = flatten_metrics(metrics)
        if flat:
            mlflow.log_metrics(flat)
        if config:
            params = {}
            for k, v in flatten_metrics(config).items():
                params[k] = v
            # keep a few string fields
            for key in ("model", "protocol", "segment", "device"):
                if key in (config or {}) and config[key] is not None:
                    params[key] = str(config[key])
            if "model" in config and isinstance(config["model"], dict):
                for mk in ("name", "weights", "imgsz", "conf"):
                    if mk in config["model"]:
                        params[f"model.{mk}"] = str(config["model"][mk])
            if params:
                mlflow.log_params({k: str(v)[:250] for k, v in list(params.items())[:100]})
        if run_dir is not None:
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                mlflow.log_artifact(str(metrics_file))
        info = mlflow.active_run()
        run_id = info.info.run_id if info else None
        print("mlflow: uri=", uri, "run_id=", run_id)
        return run_id


def log_remote(
    metrics: dict[str, Any],
    config: dict[str, Any] | None,
    logging_cfg: dict[str, Any],
    run_dir: Path | None = None,
) -> dict[str, str | None]:
    urls: dict[str, str | None] = {"wandb": None, "mlflow": None}
    wandb_cfg = logging_cfg.get("wandb") or {}
    mlflow_cfg = logging_cfg.get("mlflow") or {}
    run_name = logging_cfg.get("run_name")

    if wandb_cfg.get("enabled", True):
        urls["wandb"] = log_wandb(
            metrics,
            config,
            project=str(wandb_cfg.get("project") or "pose-av"),
            run_name=run_name,
            entity=wandb_cfg.get("entity"),
            run_dir=run_dir,
        )
    if mlflow_cfg.get("enabled"):
        urls["mlflow"] = log_mlflow(
            metrics,
            config,
            tracking_uri=str(mlflow_cfg.get("tracking_uri") or ""),
            experiment=str(mlflow_cfg.get("experiment") or "pose-av"),
            run_name=run_name,
            run_dir=run_dir,
        )
    return urls
