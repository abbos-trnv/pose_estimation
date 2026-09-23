# ГП2 на Kaggle

Ноутбук: `notebooks/gp2_baseline.ipynb`.  
Срез уже на диске (один сегмент, ~330 MB картинок):

- `data/waymo_v2/training/camera_image/10023947602400723454_1120_000_1140_000.parquet`
- `data/waymo_v2/training/camera_box/10023947602400723454_1120_000_1140_000.parquet`
- `data/waymo_v2/training/camera_hkp/camera_hkp/10023947602400723454_1120_000_1140_000.parquet` (497 GT pose)

## Загрузка датасета

1. Готовый архив: `data/kaggle_gp2_slice.zip` (~327 MB). Либо папка `data/kaggle_gp2_slice/training/...`.
2. Kaggle → Dataset → New → upload **этот zip** (private).
3. New Notebook → GPU T4 → Add data → этот датасет.
4. Upload notebook `gp2_baseline.ipynb` (или File → Import).
5. Run All.

`MAX_FRAMES = 40` для быстрого прогона; `None` — весь сегмент (~995 кадров, дольше).

После Run All скопируй JSON `summary` (mean_OKS, PCK@0.2, latency, unmatched_gt) — это цифры в отчёт ГП2.

Код на Kaggle **не грузим zip-ом и не Upload File**. Репо публичный:

```
git clone --depth 1 -b missing-ml-2026 https://github.com/abbos-trnv/pose_estimation.git
```

Ноутбук `gp4_train.ipynb` это делает сам (Internet On). После `git push` на GitHub достаточно Run All — подтянется новая версия. Zip с картинками заливается **один раз**.


Не меряй это на CPU дома: YOLO на 40 кадрах на Kaggle T4 — минуты, локально будет долго.
