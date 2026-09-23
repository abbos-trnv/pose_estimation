"""Sanity-check OKS/PCK on synthetic poses (no dataset download)."""

import numpy as np

from pose_av.metrics import coco17_to_waymo14, mean_oks, pck


def _perfect_batch(n: int = 4, k: int = 14):
    rng = np.random.default_rng(0)
    gt = rng.uniform(10, 90, size=(n, k, 2))
    vis = np.ones((n, k), dtype=np.int32)
    vis[0, 0] = 0
    box = np.full((n, 2), 80.0)
    return gt, vis, box


def test_oks_perfect_is_one():
    gt, vis, box = _perfect_batch()
    assert mean_oks(gt, gt.copy(), vis, box) == 1.0


def test_oks_far_prediction_is_low():
    gt, vis, box = _perfect_batch()
    pr = gt + 400.0
    assert mean_oks(gt, pr, vis, box) < 0.05


def test_pck_perfect_is_one():
    gt, vis, box = _perfect_batch()
    assert pck(gt, gt.copy(), vis, box, threshold=0.2) == 1.0


def test_coco17_mapping_nose_and_shoulders():
    coco_xy = np.zeros((1, 17, 2))
    coco_vis = np.ones((1, 17), dtype=np.int32)
    coco_xy[0, 0] = [1, 2]
    coco_xy[0, 5] = [3, 4]
    coco_xy[0, 3] = [10, 0]
    coco_xy[0, 4] = [20, 0]
    waymo, vis = coco17_to_waymo14(coco_xy, coco_vis)
    assert waymo.shape == (1, 14, 2)
    np.testing.assert_array_equal(waymo[0, 0], [1, 2])
    np.testing.assert_array_equal(waymo[0, 1], [3, 4])
    np.testing.assert_array_equal(waymo[0, 13], [15, 0])
    assert vis[0, 13] == 1
