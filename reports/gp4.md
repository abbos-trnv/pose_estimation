# Детекция ключевых точек пешеходов в сценах автономного вождения

**Отчёт по вехе 4 (ГП4): экспериментирование**

Тургунов Аббос  
НИУ ВШЭ, 23 сентября 2026 г.  
Репозиторий: `pose_estimation/` ветка `missing-ml-2026`

---

## 1. Что требует веха

- параллелизация обучения;
- сетка экспериментов и **новый скор** относительно baseline ГП2 (PCK@0.2 / покрытие, не только mean OKS).

Baseline ГП2 — COCO YOLOv8s-pose **без** обучения, 40 кадров. ГП4: fine-tune на **всех кадрах сегмента**, val — хвост по времени.

---

## 2. Пайплайн

1. `scripts/export_yolo_pose.py` — Waymo-14 → COCO-17 YOLO labels (`data/yolo_pose_waymo/`). Глаза не размечаем; уши = лоб, если он есть.  
2. `scripts/train_pose.py` — Ultralytics fine-tune поверх COCO.  
3. `scripts/eval_pose.py model.weights=.../best.pt` — те же протоколы + срез **small/medium/large** по площади GT-бокса.

Ноутбук: `notebooks/gp4_train.ipynb`.

---

## 3. Сетка

| id | Что | Зачем |
|---|---|---|
| E0 | ГП2 `yolov8s-pose.pt`, 40 кадров | якорь |
| E1 | s-pose, fine-tune, `imgsz=640`, AMP, workers=4 | student на домене |
| E2 | n-pose, то же | скорость |
| E3 | s-pose, `imgsz=1280` | мелкие люди |
| E4 | eval `conf` 0.25 vs 0.35 vs 0.5 | recall vs FP |

Hydra:

```text
python scripts/export_yolo_pose.py
python scripts/train_pose.py
python scripts/eval_pose.py data=waymo_full data.subset=val model.weights=runs/train/gp4_s1280/weights/best.pt
```

На Kaggle 2×T4: `train.device=[0,1]` (DDP в Ultralytics).

---

## 4. Параллелизация (как в лекции / ГП4 других групп)

На нашей модели (YOLO-pose, мало параметров) осмысленно:

- **AMP** `train.amp=true` — дефолт;
- **DataLoader workers** `train.workers=4`;
- **DDP** `device=[0,1]` если Kaggle T4×2;
- **Hydra multirun** `-m` — сетка без правки кода.

Gradient checkpointing не нужен (модель влезает в 16 GB).

---

## 5. Скор

Рабочий прогон — **100 эпох, imgsz=1280**, eval **только val**. Короткий 20 эпох @ 640 в отчёт не идёт.

Якорь ГП2 (COCO, 40 кадров): crop PCK@0.2 = 0.435, full-frame unmatched = 100.

Цель: crop PCK заметно выше 0.45 на val; small-бокс PCK не 0.28; full-frame unmatched ниже доли COCO на том же val.

---

## 6. Команды (Kaggle, несколько часов)

```text
python scripts/export_yolo_pose.py
python scripts/train_pose.py
python scripts/eval_pose.py data=waymo_full data.subset=val model.weights=runs/train/gp4_s1280/weights/best.pt
```

OOM: `train.batch=2`. Не выключать ноутбук, пока идёт train.
