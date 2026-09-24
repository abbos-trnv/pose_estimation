# Данные

Картинки Waymo в git не хранятся.

| Срез | Зачем |
|---|---|
| GP2 zip (~327 MB) | один сегмент Perception для baseline |
| `waymo_v2/training` | parquet боксов, keypoints и кадров (локально / Kaggle) |
| `yolo_pose_waymo` | jpeg + txt для дообучения YOLO |

Версия среза GP2 фиксируется DVC (`kaggle_gp2_slice.zip.dvc`): в git только хеш, сами байты — в кэше DVC или рядом на диске.

Повтор baseline: тот же сегмент `10023947602400723454_1120_000_1140_000`, 40 кадров (конфиг `waymo_slice`).
