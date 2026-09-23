"""GP2 protocols: full-frame bottom-up vs GT-crop top-down."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from pose_av.boxes import center_wh_to_xyxy, match_greedy
from pose_av.data_waymo import SliceTables, gt_keypoints
from pose_av.metrics import mean_oks, pck


class PoseBackend(Protocol):
    def predict_full_frame(self, image_bgr: np.ndarray) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], float]:
        """boxes xyxy, (kpt_xy, kpt_vis) Waymo-14, latency seconds."""

    def predict_crop(self, image_bgr: np.ndarray, xyxy: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
        """Waymo-14 keypoints in full-image coords, or None."""


@dataclass
class ProtocolStats:
    name: str
    oks: list[float] = field(default_factory=list)
    pck: list[float] = field(default_factory=list)
    unmatched: int = 0
    latency_s: list[float] = field(default_factory=list)

    def as_dict(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "protocol": self.name,
            "matched_instances": len(self.oks),
            "unmatched_gt": self.unmatched,
            "mean_OKS": float(np.nanmean(self.oks)) if self.oks else None,
            "PCK@0.2": float(np.nanmean(self.pck)) if self.pck else None,
        }
        if self.latency_s:
            mean_dt = float(np.mean(self.latency_s))
            out["latency_ms_mean"] = mean_dt * 1000.0
            out["fps"] = 1.0 / mean_dt if mean_dt > 0 else None
        if extra:
            out.update(extra)
        return out


def decode_jpeg(blob: bytes) -> np.ndarray | None:
    import cv2

    arr = np.frombuffer(blob, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return im


def run_protocols(
    slice_data: SliceTables,
    backend: PoseBackend,
    *,
    iou_match: float = 0.3,
    do_full_frame: bool = True,
    do_gt_crop: bool = True,
    progress_every: int = 10,
) -> dict[str, Any]:
    ff = ProtocolStats("full_frame")
    td = ProtocolStats("gt_crop")
    t_loop = time.perf_counter()

    for fi, key in enumerate(slice_data.keys):
        im = decode_jpeg(slice_data.images[key])
        if im is None:
            continue

        p_boxes: list[np.ndarray] = []
        p_kpts: list[tuple[np.ndarray, np.ndarray]] = []
        if do_full_frame:
            p_boxes, p_kpts, dt = backend.predict_full_frame(im)
            ff.latency_s.append(dt)

        box_by_oid = {b["oid"]: b for b in slice_data.boxes.get(key, [])}
        for rec in slice_data.hkps.get(key, []):
            b = box_by_oid.get(rec["oid"])
            if b is None:
                if do_full_frame:
                    ff.unmatched += 1
                if do_gt_crop:
                    td.unmatched += 1
                continue
            gt_xyxy = center_wh_to_xyxy(b["cx"], b["cy"], b["w"], b["h"])
            gt_xy, gt_vis = gt_keypoints(rec)
            if gt_vis.sum() == 0:
                continue
            wh = np.array([b["w"], b["h"]], dtype=np.float64)

            if do_full_frame:
                best_i = match_greedy(gt_xyxy, p_boxes, iou_match)
                if best_i < 0:
                    ff.unmatched += 1
                else:
                    pr_xy, _ = p_kpts[best_i]
                    ff.oks.append(mean_oks(gt_xy[None], pr_xy[None], gt_vis[None], wh[None]))
                    ff.pck.append(pck(gt_xy[None], pr_xy[None], gt_vis[None], wh[None], threshold=0.2))

            if do_gt_crop:
                crop_pred = backend.predict_crop(im, gt_xyxy)
                if crop_pred is None:
                    td.unmatched += 1
                else:
                    pr_xy, _ = crop_pred
                    td.oks.append(mean_oks(gt_xy[None], pr_xy[None], gt_vis[None], wh[None]))
                    td.pck.append(pck(gt_xy[None], pr_xy[None], gt_vis[None], wh[None], threshold=0.2))

        if progress_every and (fi + 1) % progress_every == 0:
            print(f"{fi + 1}/{len(slice_data.keys)}  ff_matched={len(ff.oks)} td_matched={len(td.oks)}")

    summary: dict[str, Any] = {
        "frames": len(slice_data.keys),
        "elapsed_s": time.perf_counter() - t_loop,
    }
    if do_full_frame:
        summary["full_frame_bottom_up"] = ff.as_dict()
    if do_gt_crop:
        summary["gt_crop_top_down"] = td.as_dict()
    return summary
