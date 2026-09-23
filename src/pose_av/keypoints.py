"""Waymo camera skeleton (14 joints) and mapping from COCO-17."""

from __future__ import annotations

from dataclasses import dataclass

# Proto numbers from waymo_open_dataset/protos/keypoint.proto
NOSE = 1
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 13
LEFT_ELBOW = 6
RIGHT_ELBOW = 14
LEFT_WRIST = 7
RIGHT_WRIST = 15
LEFT_HIP = 8
RIGHT_HIP = 16
LEFT_KNEE = 9
RIGHT_KNEE = 17
LEFT_ANKLE = 10
RIGHT_ANKLE = 18
FOREHEAD = 19
HEAD_CENTER = 20

# CANONICAL_ORDER_CAMERA in waymo_open_dataset.utils.keypoint_data
CAMERA_ORDER: tuple[int, ...] = (
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    FOREHEAD,
)

CAMERA_NAMES: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "forehead",
)

# DEFAULT_PER_TYPE_SCALES from waymo_open_dataset.metrics.python.keypoint_metrics
PER_TYPE_SCALES: dict[int, float] = {
    NOSE: 0.052,
    LEFT_SHOULDER: 0.158,
    RIGHT_SHOULDER: 0.158,
    LEFT_ELBOW: 0.144,
    RIGHT_ELBOW: 0.144,
    LEFT_WRIST: 0.124,
    RIGHT_WRIST: 0.124,
    LEFT_HIP: 0.214,
    RIGHT_HIP: 0.214,
    LEFT_KNEE: 0.174,
    RIGHT_KNEE: 0.174,
    LEFT_ANKLE: 0.178,
    RIGHT_ANKLE: 0.178,
    FOREHEAD: 0.158,
    HEAD_CENTER: 0.158,
}

CAMERA_SCALES: tuple[float, ...] = tuple(PER_TYPE_SCALES[t] for t in CAMERA_ORDER)

# COCO-17: nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_elb, r_elb,
# l_wri, r_wri, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank
COCO17_TO_WAYMO_CAMERA: dict[int, int | tuple[int, int]] = {
    0: 0,  # nose -> nose
    5: 1,  # l_sh
    6: 2,  # r_sh
    7: 3,  # l_elb
    8: 4,  # r_elb
    9: 5,  # l_wri
    10: 6,  # r_wri
    11: 7,  # l_hip
    12: 8,  # r_hip
    13: 9,  # l_knee
    14: 10,  # r_knee
    15: 11,  # l_ank
    16: 12,  # r_ank
    # forehead ≈ midpoint of ears (COCO 3, 4)
    "forehead": (3, 4),
}

SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 13),  # nose-forehead
    (1, 2),  # shoulders
    (1, 3),
    (3, 5),
    (2, 4),
    (4, 6),
    (1, 7),
    (2, 8),
    (7, 8),
    (7, 9),
    (9, 11),
    (8, 10),
    (10, 12),
)

OKS_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
PCK_THRESHOLDS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


@dataclass(frozen=True)
class DatasetCatalogEntry:
    name: str
    keypoints: str
    download_slice: str
    role: str
