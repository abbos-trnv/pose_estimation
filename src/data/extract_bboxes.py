"""
Extract 2D bounding boxes from nuScenes 3D annotations.

Projection chain: world -> ego -> camera -> image
Uses nuscenes Box + pyquaternion for correct coordinate transforms.
"""
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def project_3d_box_to_2d(nusc, ann_record, sample_data_record, img_w=1600, img_h=900):
    """
    Project 3D bounding box to 2D image coordinates.

    Steps:
        1. Create Box in world coordinates
        2. Transform world -> ego (car frame)
        3. Transform ego -> camera frame
        4. Project 8 corners onto image with camera intrinsic
        5. Take min/max of projected corners -> 2D bbox

    Returns:
        [x1, y1, x2, y2] or None if box is behind camera or outside image.
    """
    # 1. Create Box in world coordinates
    box = Box(
        ann_record['translation'],
        ann_record['size'],
        Quaternion(ann_record['rotation']),
    )

    # 2. World -> ego
    ego = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
    box.translate(-np.array(ego['translation']))
    box.rotate(Quaternion(ego['rotation']).inverse)

    # 3. Ego -> camera
    cs = nusc.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
    box.translate(-np.array(cs['translation']))
    box.rotate(Quaternion(cs['rotation']).inverse)

    # 4. Skip if behind camera
    if box.center[2] <= 0:
        return None

    # 5. Project 8 corners to image
    K = np.array(cs['camera_intrinsic'])
    corners_3d = box.corners()  # shape (3, 8)
    corners_2d = view_points(corners_3d, K, normalize=True)  # shape (3, 8)

    # 6. Get 2D bbox from projected corners
    x1 = max(0, int(np.min(corners_2d[0, :])))
    y1 = max(0, int(np.min(corners_2d[1, :])))
    x2 = min(img_w, int(np.max(corners_2d[0, :])))
    y2 = min(img_h, int(np.max(corners_2d[1, :])))

    if x2 <= x1 or y2 <= y1:
        return None

    return [float(x1), float(y1), float(x2), float(y2)]


def extract_bboxes_for_sample(nusc, sample_token, camera='CAM_FRONT'):
    """Extract 2D bboxes for all pedestrians in a sample for a given camera."""
    sample = nusc.get('sample', sample_token)

    if camera not in sample['data']:
        return [], ''

    sd = nusc.get('sample_data', sample['data'][camera])

    bboxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        instance = nusc.get('instance', ann['instance_token'])
        cat = nusc.get('category', instance['category_token'])

        if 'pedestrian' not in cat['name']:
            continue

        bbox_2d = project_3d_box_to_2d(nusc, ann, sd)
        if bbox_2d is None:
            continue

        bboxes.append({
            'bbox': bbox_2d,
            'category': cat['name'],
            'ann_token': ann_token,
            'instance_token': ann['instance_token'],
            'translation_3d': ann['translation'],
            'size_3d': ann['size'],
            'num_lidar_pts': ann.get('num_lidar_pts', 0),
        })

    return bboxes, sd['filename']


if __name__ == '__main__':
    nusc = NuScenes(
        version='v1.0-trainval',
        dataroot='data/images/v1.0-trainval_meta',
        verbose=False,
    )

    # Test on first sample with pedestrians
    for sample in nusc.sample:
        bboxes, filename = extract_bboxes_for_sample(nusc, sample['token'])
        if bboxes:
            print('Filename:', filename)
            print('Found', len(bboxes), 'pedestrians')
            for b in bboxes[:3]:
                print('  bbox:', [int(x) for x in b['bbox']])
                print('  category:', b['category'])
            break