# Запуск обучения локально (Mac MPS)

## Данные уже готовы:
- `data/yolo_pose/images/*.txt` - 17,021 файл с keypoints
- `data/yolo_pose/splits/` - train/val/test
- `data/images/` - изображения (15GB)

## Запуск

```bash
.venv/bin/python scripts/train_local.py
```

## Параметры по умолчанию:
- epochs: 50
- batch: 16 (на MPS)
- device: mps

## После обучения

Модель сохранится в:
```
runs/pose/yolov8n_nuscenes/weights/best.pt
```

## Оценка

```bash
.venv/bin/python -c "
from ultralytics import YOLO
model = YOLO('runs/pose/yolov8n_nuscenes/weights/best.pt')
metrics = model.val(data='data/yolo_pose/nuscenes_pose.yaml', split='test')
print(f'Pose mAP50: {metrics.pose.map50}')
print(f'Pose mAP50-95: {metrics.pose.map}')
"
```