"""OKS / PCK for Waymo-style 2D keypoints. Independent of TensorFlow."""

from __future__ import annotations

import numpy as np

from pose_av.keypoints import CAMERA_SCALES


def object_scale(width: np.ndarray, height: np.ndarray) -> np.ndarray:
    """sqrt(area), same convention as Waymo 2D box.scale."""
    return np.sqrt(np.maximum(width * height, 1e-12))


def oks_per_instance(
    gt_xy: np.ndarray,
    pr_xy: np.ndarray,
    visibility: np.ndarray,
    box_wh: np.ndarray,
    per_type_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Mean OKS over visible GT joints.

    gt_xy, pr_xy: [N, K, 2]
    visibility: [N, K]  — 0 = unlabeled/absent, >0 = use in metric
    box_wh: [N, 2] width, height in the same units as xy
    """
    if per_type_scales is None:
        per_type_scales = np.asarray(CAMERA_SCALES, dtype=np.float64)
    else:
        per_type_scales = np.asarray(per_type_scales, dtype=np.float64)

    scale = object_scale(box_wh[:, 0], box_wh[:, 1])[:, None]  # [N, 1]
    denom = 2.0 * (per_type_scales[None, :] * scale) ** 2
    delta = gt_xy - pr_xy
    dist2 = np.sum(delta * delta, axis=-1)
    vis = visibility > 0
    oks_k = np.exp(-dist2 / np.maximum(denom, 1e-12))
    oks_k = np.where(vis, oks_k, 0.0)
    counts = vis.sum(axis=1).astype(np.float64)
    return np.divide(oks_k.sum(axis=1), np.maximum(counts, 1.0))


def mean_oks(
    gt_xy: np.ndarray,
    pr_xy: np.ndarray,
    visibility: np.ndarray,
    box_wh: np.ndarray,
    per_type_scales: np.ndarray | None = None,
) -> float:
    values = oks_per_instance(gt_xy, pr_xy, visibility, box_wh, per_type_scales)
    has = (visibility > 0).any(axis=1)
    if not np.any(has):
        return float("nan")
    return float(values[has].mean())


def pck(
    gt_xy: np.ndarray,
    pr_xy: np.ndarray,
    visibility: np.ndarray,
    box_wh: np.ndarray,
    threshold: float = 0.2,
    per_type_scales: np.ndarray | None = None,
) -> float:
    """PCK with object-relative threshold: t * k_i * sqrt(area)."""
    if per_type_scales is None:
        per_type_scales = np.asarray(CAMERA_SCALES, dtype=np.float64)
    else:
        per_type_scales = np.asarray(per_type_scales, dtype=np.float64)

    scale = object_scale(box_wh[:, 0], box_wh[:, 1])[:, None]
    abs_thr = threshold * per_type_scales[None, :] * scale
    dist = np.linalg.norm(gt_xy - pr_xy, axis=-1)
    vis = visibility > 0
    correct = (dist <= abs_thr) & vis
    total = vis.sum()
    if total == 0:
        return float("nan")
    return float(correct.sum() / total)


def coco17_to_waymo14(coco_xy: np.ndarray, coco_vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map COCO-17 [..., 17, 2] onto Waymo camera-14.

    Forehead is the midpoint of the two ears when both are visible, else nose.
    """
    *batch, _, _xy = coco_xy.shape
    waymo = np.zeros((*batch, 14, 2), dtype=coco_xy.dtype)
    vis = np.zeros((*batch, 14), dtype=coco_vis.dtype)
    coco_idx = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    for dst, src in enumerate(coco_idx):
        waymo[..., dst, :] = coco_xy[..., src, :]
        vis[..., dst] = coco_vis[..., src]

    ears_xy = 0.5 * (coco_xy[..., 3, :] + coco_xy[..., 4, :])
    ears_ok = (coco_vis[..., 3] > 0) & (coco_vis[..., 4] > 0)
    waymo[..., 13, :] = np.where(ears_ok[..., None], ears_xy, coco_xy[..., 0, :])
    vis[..., 13] = np.where(ears_ok, np.minimum(coco_vis[..., 3], coco_vis[..., 4]), coco_vis[..., 0])
    return waymo, vis


def waymo14_to_coco17(waymo_xy: np.ndarray, waymo_vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of coco17_to_waymo14 for YOLO-pose training labels.

    Eyes stay unlabeled. Ears copy the forehead when it is visible.
    """
    *batch, _, _xy = waymo_xy.shape
    coco_xy = np.zeros((*batch, 17, 2), dtype=waymo_xy.dtype)
    coco_vis = np.zeros((*batch, 17), dtype=waymo_vis.dtype)
    coco_xy[..., 0, :] = waymo_xy[..., 0, :]
    coco_vis[..., 0] = waymo_vis[..., 0]
    coco_xy[..., 5:17, :] = waymo_xy[..., 1:13, :]
    coco_vis[..., 5:17] = waymo_vis[..., 1:13]
    coco_xy[..., 3, :] = waymo_xy[..., 13, :]
    coco_xy[..., 4, :] = waymo_xy[..., 13, :]
    coco_vis[..., 3] = waymo_vis[..., 13]
    coco_vis[..., 4] = waymo_vis[..., 13]
    return coco_xy, coco_vis
