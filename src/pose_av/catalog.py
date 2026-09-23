"""Published catalog facts used in GP1 until a local Waymo slice is downloaded."""

from __future__ import annotations

from pose_av.keypoints import DatasetCatalogEntry

CATALOG: tuple[DatasetCatalogEntry, ...] = (
    DatasetCatalogEntry(
        name="Waymo Perception v2",
        keypoints="14 joints, hand-labeled; 172.6k camera / 10k laser objects",
        download_slice="parquet components: camera_hkp ~45 MB, camera_image ~387 GB",
        role="primary GT and eval",
    ),
    DatasetCatalogEntry(
        name="nuScenes / nuImages",
        keypoints="no skeleton; pose = standing/sitting/moving",
        download_slice="v1.0-mini (10 scenes); full trainval is hundreds of GB",
        role="pseudo-GT / AV domain extras",
    ),
    DatasetCatalogEntry(
        name="nuPlan",
        keypoints="no official skeleton (community SMPL only)",
        download_slice="planner labels; 10% sensors ~16 TB",
        role="out of GP1–GP4 scope",
    ),
    DatasetCatalogEntry(
        name="WOSAC / WOMD",
        keypoints="none; agents are boxes for 8 s @ 10 Hz",
        download_slice="scenario ~646 GB",
        role="out of GP1–GP4 scope",
    ),
    DatasetCatalogEntry(
        name="PedX",
        keypoints="18 2D joints + automatic SMPL 3D",
        download_slice="~5k stereo pairs, 14k pedestrian instances",
        role="optional external slice",
    ),
)

WAYMO_PUBLIC = {
    "camera_keypoint_objects": 172_600,
    "laser_keypoint_objects": 10_000,
    "joints": 14,
    "camera_image_gb": 386.735,
    "camera_hkp_gb": 0.045,
    "lidar_hkp_gb": 0.016,
    "full_perception_v2_gb": 682.62,
}

NUSCENES_PRIOR_FILTER = {
    "raw_pedestrian_boxes": 265_000,
    "after_geometry_filters": 94_000,
    "after_min_lidar_pts": 41_000,
    "images_with_pseudo_gt": 17_739,
    "pedestrians_with_17_kpts": 41_492,
    "split_train": 12_266,
    "split_val": 3_688,
    "split_test": 1_785,
    "split_unit": "images, scene-level",
}
