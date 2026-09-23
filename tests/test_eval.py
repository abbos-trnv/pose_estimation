"""Box matching and mocked eval loop (no YOLO, no Waymo download)."""

import cv2
import numpy as np

from pose_av.boxes import iou_xyxy, match_greedy
from pose_av.data_waymo import SliceTables, gt_keypoints
from pose_av.evaluate import run_protocols
from pose_av.keypoints import CAMERA_ORDER


def test_iou_identical_is_one():
    box = [10, 10, 30, 40]
    assert iou_xyxy(box, box) == 1.0


def test_match_greedy_picks_overlap():
    gt = [0, 0, 10, 10]
    preds = [np.array([50, 50, 60, 60]), np.array([1, 1, 11, 11])]
    assert match_greedy(gt, preds, iou_thr=0.3) == 1
    assert match_greedy(gt, preds[:1], iou_thr=0.3) == -1


class FakeBackend:
    def predict_full_frame(self, image_bgr):
        boxes = [np.array([0.0, 0.0, 20.0, 40.0])]
        xy = np.zeros((14, 2))
        vis = np.ones(14)
        return boxes, [(xy, vis)], 0.01

    def predict_crop(self, image_bgr, xyxy):
        return np.zeros((14, 2)), np.ones(14)


def test_run_protocols_with_fake_backend():
    rec = {
        "oid": "a",
        "types": list(CAMERA_ORDER),
        "x": [1.0] * 14,
        "y": [2.0] * 14,
        "occ": [False] * 14,
    }
    im = np.zeros((48, 48, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", im)
    assert ok
    slice_data = SliceTables(
        images={(0, 1): buf.tobytes()},
        boxes={(0, 1): [{"oid": "a", "cx": 10.0, "cy": 20.0, "w": 20.0, "h": 40.0}]},
        hkps={(0, 1): [rec]},
        keys=[(0, 1)],
        paths={},
    )
    summary = run_protocols(slice_data, FakeBackend(), progress_every=0)
    assert summary["full_frame_bottom_up"]["matched_instances"] == 1
    assert summary["gt_crop_top_down"]["matched_instances"] == 1


def test_gt_keypoints_order():
    rec = {"types": [1, 5], "x": [3.0, 8.0], "y": [4.0, 9.0], "occ": [False, False]}
    xy, vis = gt_keypoints(rec)
    assert vis[0] == 1
    np.testing.assert_array_equal(xy[0], [3.0, 4.0])
    np.testing.assert_array_equal(xy[1], [8.0, 9.0])
