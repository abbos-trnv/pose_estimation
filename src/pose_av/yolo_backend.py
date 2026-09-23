"""Ultralytics YOLO-pose wrapper. Optional dependency (eval only)."""

from __future__ import annotations

import time

import numpy as np

from pose_av.metrics import coco17_to_waymo14


class YoloPoseBackend:
    def __init__(
        self,
        weights: str,
        device: str,
        imgsz: int,
        conf: float,
        crop_imgsz: int = 256,
        crop_conf: float = 0.1,
        kpt_conf: float = 0.2,
        crop_kpt_conf: float = 0.15,
        crop_pad: float = 0.2,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.crop_imgsz = crop_imgsz
        self.crop_conf = crop_conf
        self.kpt_conf = kpt_conf
        self.crop_kpt_conf = crop_kpt_conf
        self.crop_pad = crop_pad

    def predict_full_frame(
        self, image_bgr: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], float]:
        t0 = time.perf_counter()
        pred = self.model.predict(
            image_bgr,
            verbose=False,
            device=self.device,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[0],
        )[0]
        dt = time.perf_counter() - t0
        p_boxes: list[np.ndarray] = []
        p_kpts: list[tuple[np.ndarray, np.ndarray]] = []
        if pred.boxes is not None and pred.keypoints is not None and len(pred.boxes):
            xyxy = pred.boxes.xyxy.cpu().numpy()
            kxy = pred.keypoints.xy.cpu().numpy()
            if pred.keypoints.conf is not None:
                kconf = pred.keypoints.conf.cpu().numpy()
            else:
                kconf = np.ones(kxy.shape[:2])
            for bi in range(len(xyxy)):
                vis = (kconf[bi] > self.kpt_conf).astype(np.float64)
                p_boxes.append(xyxy[bi])
                p_kpts.append(coco17_to_waymo14(kxy[bi], vis))
        return p_boxes, p_kpts, dt

    def predict_crop(self, image_bgr: np.ndarray, xyxy: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        pad_x = int(self.crop_pad * (x2 - x1))
        pad_y = int(self.crop_pad * (y2 - y1))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return None
        crop = image_bgr[y1:y2, x1:x2]
        pred = self.model.predict(
            crop,
            verbose=False,
            device=self.device,
            conf=self.crop_conf,
            imgsz=self.crop_imgsz,
            classes=[0],
        )[0]
        if pred.keypoints is None or len(pred.keypoints) == 0:
            return None
        kxy = pred.keypoints.xy.cpu().numpy()[0].copy()
        if pred.keypoints.conf is not None:
            kconf = pred.keypoints.conf.cpu().numpy()[0]
        else:
            kconf = np.ones(17)
        kxy[:, 0] += x1
        kxy[:, 1] += y1
        vis = (kconf > self.crop_kpt_conf).astype(np.float64)
        return coco17_to_waymo14(kxy, vis)
