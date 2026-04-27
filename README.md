# Pose Estimation for Autonomous Driving

Детекция пешеходов и их ключевых точек (pose estimation) для систем автономного вождения.

## Подход: Knowledge Distillation

```
Teacher (ViTPose-base) → псевдо-лейблы → Student (YOLOv8n-pose)
```

1. **ViTPose-base** (teacher) - генерирует псевдо-лейблы на nuScenes
2. **YOLOv8n-pose** (student) - обучается на псевдо-лейблах
3. **Цель** - сжатие до ~100ms latency на CPU

## Датасет

- **nuScenes** (3/9 keyframes)
- **17,739 изображений** с **41,492 пешеходами**
- 17 ключевых точек COCO формат

### Разделение по сценам
- Train: 12,266 изображений
- Val: 3,688 изображений  
- Test: 1,785 изображений

## Фильтрация данных

```python
min_lidar_pts = 5       # минимум точек лидара
min_bbox_width = 20      # минимальная ширина
min_bbox_height = 50     # минимальная высота
```

## Результаты (ViTPose)

| Метрика | Значение |
|---------|----------|
| Изображений | 17,739 |
| Пешеходов | 41,492 |

## Структура проекта

```
├── data/
│   ├── annotations/
│   │   ├── bboxes.json      # 3D→2D bbox пешеходов
│   │   └── keypoints.json # псевдо-лейблы ViTPose
│   └── yolo_pose/        # YOLO формат для обучения
├── scripts/
│   ├── convert_coco_to_yolo.py  # Конвертация в YOLO формат
│   ├── split_dataset.py          # Разделение train/val/test
│   └── train_local.py           # Обучение локально
├── notebooks/
│   └── colab_direct.py         # Инструкция для Colab
├── weights/
│   └── yolov8n-pose.pt     # Стартовая модель
└── Отчет.md                  # Подробный отчет
```

## Обучение Student модели

### Локально (Mac MPS)
```bash
.venv/bin/python scripts/train_local.py
```

### В Google Colab
1. Загрузить `yolo_pose.zip` на Google Drive
2. Скачать изображения nuScenes (или использовать локальные)
3. Запустить `notebooks/colab_direct.py`

### Параметры обучения
- model: YOLOv8n-pose (3.3M params)
- epochs: 50-100
- batch: 16-32
- device: MPS / CUDA
- optimizer: AdamW

## Модели

| Модель | Параметры | Latency (CPU) | AP (COCO) |
|--------|-----------|--------------|-----------|
| YOLOv8n-pose | 3.3M | ~20ms | ~50 |
| YOLOv8s-pose | 11.6M | ~50ms | ~58 |
| ViTPose-base | 86M | ~500ms | ~70 |

## Требования

- Python 3.11+
- PyTorch 2.0+
- ultralytics
- torchvision
- numpy, pillow

## Ссылки

- [nuScenes](https://www.nuscenes.org/)
- [ViTPose](https://huggingface.co/usyd-community/vitpose-base-simple)
- [YOLOv8](https://docs.ultralytics.com/)

