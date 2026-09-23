# Детекция ключевых точек пешеходов в сценах автономного вождения

**Отчёт по вехе 4 (ГП4): экспериментирование**

Тургунов Аббос  
Статус: **инфраструктура и сетка готовы; цифры fine-tune — после прогона на Kaggle.**  
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
python scripts/train_pose.py -m model=yolov8n_pose,yolov8s_pose train.imgsz=640
python scripts/eval_pose.py model.weights=runs/train/gp4/weights/best.pt data.max_frames=null
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

## 5. Метрики до обучения (якорь)

Срез 40 кадров, `yolov8s-pose.pt` COCO, W&B run `0yfnuwgz`:

| | matched | unmatched | OKS | PCK@0.2 |
|---|---|---|---|---|
| full_frame | 53 | 100 | 0.956 | 0.464 |
| gt_crop | 150 | 3 | 0.905 | 0.435 |

Цель E1: **PCK@0.2 выше 0.45 на gt_crop** и меньше unmatched на full_frame, когда eval на val-хвосте сегмента (не на тех же 40 train-кадрах).

Таблица E1–E4 заполняется после Kaggle.

---

## 6. Команды

```powershell
$env:PYTHONPATH="src"; $env:WAYMO_ROOT="data/waymo_v2"
pip install ultralytics torch wandb
python scripts/export_yolo_pose.py
python scripts/train_pose.py train.epochs=20
python scripts/eval_pose.py model.weights=runs/train/gp4/weights/best.pt data.max_frames=null
```
