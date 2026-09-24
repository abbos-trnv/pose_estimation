# Детекция ключевых точек пешеходов в сценах автономного вождения

**Отчёт по вехе 4 (ГП4): экспериментирование**

Тургунов Аббос  
Курс: Дополнительные главы машинного обучения / *The Missing Semester of Your ML Education*  
НИУ ВШЭ, 24 сентября 2026 г.  
Код: https://github.com/abbos-trnv/pose_estimation/tree/missing-ml-2026  
W&B: https://wandb.ai/abbostrnv178-hse-university/pose-av

---

## 1. Постановка

Target: 2D keypoints пешехода на кадре камеры. GT — Waymo `camera_hkp`, **14 суставов** (нос, плечи, локти, запястья, бёдра, колени, лодыжки, лоб). YOLO учится в COCO-17; маппинг в `src/pose_av/metrics.py`.

Модель: **YOLOv8s-pose**, fine-tune поверх COCO. Учим детект+позу на целом кадре. Кроп по GT-боксу — протокол оценки позы при известном человеке.

KPI: **PCK@0.2** и доля unmatched GT; mean OKS — вспомогательно (на крупных боксах завышен). Срезы small/medium/large по площади бокса.

---

## 2. Данные и сплиты

| Этап | Данные |
|---|---|
| GT таблица | train `camera_hkp`: 146 002 объекта, 439 nonempty-сегментов |
| Картинки на экспериментах | **1 сегмент** `10023947602400723454_1120_000_1140_000` (~110 кадров с image+hkp) |
| E0 eval | первые 40 кадров |
| E2 eval | последние 20% того же клипа (22 кадра, `data.subset=val`) |

Вывод экспериментов: 88 картинок одной улицы недостаточно, чтобы обойти COCO по PCK. Скачаны ещё **6 nonempty-сегментов** image+box (`configs/data/gp4_segments.yaml`, ~2–5k объектов/клип). Val дальше — отдельный сегмент, не хвост того же видео. Дообучить на них до дедлайна не успели.

---

## 3. Параллелизация

- AMP (`train.amp=true`)
- DataLoader `workers=2–4`
- Kaggle T4, batch 4 @ 1280
- Hydra `-m` для сетки n/s и `imgsz`
- DDP `device=[0,1]` на 2×T4, в прогоне не включали (одна карта)

---

## 4. Запущенные эксперименты

| id | Что | Веса | Eval | W&B |
|---|---|---|---|---|
| E0 | COCO, без FT | `yolov8s-pose.pt` | 40 кадров, imgsz 1280 | [0yfnuwgz](https://wandb.ai/abbostrnv178-hse-university/pose-av/runs/0yfnuwgz) |
| E1 | FT 20 эпох, imgsz **640** | `runs/train/gp4/best.pt` | 110 кадров (train+val) | [two11lwu](https://wandb.ai/abbostrnv178-hse-university/pose-av/runs/two11lwu) |
| E2 | FT 100 эпох, imgsz **1280**, AdamW, copy-paste | `gp4_s1280/best.pt` | **val 22 кадра** | [i8j4ziwk](https://wandb.ai/abbostrnv178-hse-university/pose-av/runs/i8j4ziwk) |

E1 не сравниваем с E0 1:1 (другой набор кадров). E2 — честнее по сплиту, но val всё ещё тот же ролик.

### E0 (якорь ГП2)

| | matched | unmatched | OKS | PCK@0.2 |
|---|---|---|---|---|
| full_frame | 53 | 100 | 0.956 | 0.464 |
| gt_crop | 150 | 3 | 0.905 | **0.435** |

### E2 (боевой FT, val)

| | matched | unmatched | OKS | PCK@0.2 |
|---|---|---|---|---|
| full_frame | 77 | 10 | 0.894 | 0.369 |
| gt_crop | 68 | 19 | 0.862 | **0.350** |

Crop PCK small / medium / large: **0.22 / 0.39 / 0.44**.

**Итог по скору.** Fine-tune на одном сегменте **не выбил** crop PCK относительно COCO. Unmatched на val-хвосте меньше, но это утечка сцены, не обобщение. Узкое место — объём картинок, не число эпох.

---

## 5. Команды

```text
python scripts/export_yolo_pose.py
python scripts/train_pose.py
python scripts/eval_pose.py data=waymo_full data.subset=val model.weights=runs/train/gp4_s1280/weights/best.pt
```

Ноутбук: `notebooks/gp4_train.ipynb` (git clone ветки `missing-ml-2026`).
