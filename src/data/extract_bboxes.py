"""
Extract 2D bounding boxes from nuScenes 3D annotations.
Uses camera intrinsic matrix and ego pose to project 3D boxes to 2D.
"""
import numpy as np
from nuscenes import NuScenes


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def project_3d_box_to_2d(nusc, ann_record, sample_data_record, camera='CAM_FRONT'):
    """
    Project 3D bounding box to 2D image coordinates.
    
    Args:
        nusc: NuScenes instance
        ann_record: annotation record with 'translation' (3D center) and 'size' (W, L, H)
        sample_data_record: sample_data record for the camera
        
    Returns:
        bbox_2d: [x1, y1, x2, y2] in image coordinates, or None if behind camera
    """
    # Get camera intrinsic
    cs = nusc.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
    K = np.array(cs['camera_intrinsic'])
    
    # Get ego pose (car position in world frame)
    ego = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
    ego_trans = np.array(ego['translation'])
    ego_rot = quaternion_to_rotation_matrix(ego['rotation'])
    
    # Get annotation (box center in world frame)
    box_center = np.array(ann_record['translation'])  # [x, y, z] in world
    box_size = np.array(ann_record['size'])  # [width, length, height]
    
    # Transform box center from world to ego (car) frame
    box_in_ego = ego_rot.T @ (box_center - ego_trans)
    
    # Check if behind camera (negative z in camera frame means behind)
    if box_in_ego[0] <= 0:  # behind camera
        return None
    
    # Use simple pinhole model
    cam_x = box_in_ego[0]  # distance along optical axis
    cam_y = box_in_ego[2]  # lateral position
    cam_z = -box_in_ego[1]  # height (up)
    
    if cam_x <= 0:
        return None
    
    # Project using intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * cam_y / cam_x + cx
    v = fy * cam_z / cam_x + cy
    
    # Estimate bbox size in pixels
    distance = cam_x
    scale = fx / distance
    
    bbox_w = box_size[0] * scale * 2
    bbox_h = box_size[2] * scale * 2
    
    # Calculate corners
    x1 = max(0, int(u - bbox_w/2))
    y1 = max(0, int(v - bbox_h))
    x2 = min(1600, int(u + bbox_w/2))
    y2 = min(900, int(v))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return [float(x1), float(y1), float(x2), float(y2)]


def extract_bboxes_for_sample(nusc, sample_token, camera='CAM_FRONT'):
    """Extract 2D bboxes for all pedestrians in a sample."""
    sample = nusc.get('sample', sample_token)
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
            'num_lidar_pts': ann.get('num_lidar_pts', 0)
        })
    
    return bboxes, sd['filename']


if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/images/v1.0-trainval_meta', verbose=False)
    
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