"""Axis-aligned box helpers used to match detections to Waymo GT."""

from __future__ import annotations

import numpy as np


def iou_xyxy(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    ub = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    denom = ua + ub - inter
    return inter / denom if denom > 0 else 0.0


def center_wh_to_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def match_greedy(
    gt_xyxy: list[float],
    pred_boxes: list[np.ndarray],
    iou_thr: float,
) -> int:
    """Return index of best pred box or -1 if none above threshold."""
    best_i, best_iou = -1, iou_thr
    for i, pb in enumerate(pred_boxes):
        val = iou_xyxy(gt_xyxy, pb)
        if val > best_iou:
            best_iou, best_i = val, i
    return best_i
