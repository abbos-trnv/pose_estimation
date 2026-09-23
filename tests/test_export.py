import numpy as np

from pose_av.export_yolo import _yolo_line


def test_yolo_line_starts_with_class_and_box():
    box = {"cx": 50.0, "cy": 40.0, "w": 20.0, "h": 40.0}
    xy = np.zeros((17, 2))
    vis = np.zeros(17)
    xy[0] = [50.0, 20.0]
    vis[0] = 1
    line = _yolo_line(box, xy, vis, 100, 80)
    assert line is not None
    toks = line.split()
    assert toks[0] == "0"
    assert len(toks) == 5 + 17 * 3
