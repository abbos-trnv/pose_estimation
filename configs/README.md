# Конфиги (Hydra)

Точка входа оценки: `eval.yaml`. Обучения: `train.yaml`.

| Группа | Что задаёт |
|---|---|
| `data/waymo_slice` | один сегмент, 40 кадров (якорь ГП2) |
| `data/waymo_full` | тот же сегмент, все кадры |
| `data/gp4_segments` | train/val по разным роликам |
| `model/yolov8s_pose` | student, COCO-веса |
| `model/yolov8n_pose` | меньшая сеть |

Примеры:

```text
python scripts/eval_pose.py
python scripts/eval_pose.py model=yolov8n_pose protocol=gt_crop data.max_frames=5
python scripts/train_pose.py -m model=yolov8n_pose,yolov8s_pose
```

Seed по умолчанию 42. Логи: W&B проект `pose-av`, плюс `runs/<время>/`.
