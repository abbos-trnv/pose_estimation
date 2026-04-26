# Extract Bounding Boxes

## Что делает

Проецирует 3D bounding boxes из nuScenes в 2D bbox на изображениях.

## Как работает

1. **Входные данные:**
   - 3D bbox в мировых координатах (translation XYZ + size WHL)
   - Camera intrinsic matrix (из calibrated_sensor)
   - Ego pose (позиция машины из ego_pose)

2. **Алгоритм:**
   ```
   1. box_in_ego = R @ (box_center - ego_translation)
   2. project to 2D using: u = fx * y / x + cx, v = fy * z / x + cy
   3. estimate bbox size: pixel_size = focal_length * real_size / distance
   ```

3. **Выход:** [x1, y1, x2, y2] в пикселях

## Использование

```python
from src.data.extract_bboxes import extract_bboxes_for_sample
from nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='data/images/v1.0-trainval_meta', verbose=False)

# Для конкретного sample
bboxes, filename = extract_bboxes_for_sample(nusc, sample_token)

# filename: 'samples/CAM_FRONT/....jpg'
# bboxes: [{'bbox': [x1,y1,x2,y2], 'category': '...', ...}, ...]
```

## Тегирование

- `category`: 'human.pedestrian.adult', 'child', 'wheelchair', etc.
- `num_lidar_pts`: число точек LiDAR на объекте (0 = плохо виден)
- `translation_3d`, `size_3d`: оригинальные 3D координаты

## Ограничения

- Проекция упрощённая (pinhole model)
- Не учитывает rotation объекта
- Не учитывает occlusion

## Тест

```bash
.venv/bin/python src/data/extract_bboxes.py
```

Выводит первый кадр с пешеходом.

## Файлы

```
src/data/
├── extract_bboxes.py    # основной скрипт
└── README.md          # этот файл
```