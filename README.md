# pose-av

2D ключевые точки пешеходов в домене автономного вождения.

Курс: *The Missing Semester of Your ML Education* (HSE, 2026).  
Автор: Тургунов Аббос (соло). Закрываем **ГП1–ГП4**.

- **GT и оценка:** Waymo Perception `camera_hkp` (14 joints).
- **Пайплайн:** кадр → бокс (GT или детект) → 2D pose.
- **Не в скоупе ГП1–ГП4:** LSS/BEVFusion, планер, WOSAC как KPI.

Отчёты: [ГП1](reports/gp1.md) · [ГП2](reports/gp2.md) · [ГП3](reports/gp3.md) · [ГП4](reports/gp4.md)  
История прогонов: [HISTORY.md](HISTORY.md)

## Быстрый старт

```powershell
cd pose_estimation
python -m venv .venv
.\.venv\Scripts\pip.exe install -e .
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Для eval на картинках дополнительно:

```powershell
.\.venv\Scripts\pip.exe install ultralytics torch
```

## Eval (протокол ГП2, уже не ноутбук)

Нужен срез `camera_image` + `camera_box` + `camera_hkp` одного сегмента (см. `reports/gp2_kaggle.md`).

```powershell
$env:PYTHONPATH = "src"
$env:WAYMO_ROOT = "data/waymo_v2"
python scripts/eval_pose.py
```

Переопределение без правки файлов (Hydra):

```powershell
python scripts/eval_pose.py model=yolov8n_pose data.max_frames=5 protocol=gt_crop
python scripts/eval_pose.py model.imgsz=640 model.conf=0.25 device=cpu
```

Артефакты пишутся в `runs/<timestamp>/` (на Kaggle диск временный) **и** в облако:

- **W&B (по умолчанию):** зарегистрируйся на [wandb.ai](https://wandb.ai), `pip install wandb`, `wandb login` или секрет `WANDB_API_KEY` на Kaggle. Проект `pose-av`.
- **MLflow (опционально):** нужен tracking server (Dagshub / свой URI):

```powershell
python scripts/eval_pose.py logging.mlflow.enabled=true logging.mlflow.tracking_uri=https://dagshub.com/<user>/<repo>.mlflow
```

Без ключа скрипт не падает — пишет `wandb: skip`.

## Train (ГП4)

Нужен тот же срез картинок. Сначала лейблы YOLO-pose на **весь сегмент**, потом fine-tune, потом eval с `best.pt`:

```powershell
python scripts/export_yolo_pose.py
python scripts/train_pose.py train.epochs=20 train.amp=true train.workers=4
python scripts/eval_pose.py model.weights=runs/train/gp4/weights/best.pt data=waymo_full
```

Сетка: `python scripts/train_pose.py -m model=yolov8n_pose,yolov8s_pose`.  
Kaggle: `notebooks/gp4_train.ipynb`. Отчёт: [`reports/gp4.md`](reports/gp4.md).


## Структура

```text
configs/           Hydra: eval.yaml, data/, model/
notebooks/         EDA и зафиксированный прогон ГП2
reports/           отчёты вех
scripts/           CLI (eval_pose.py); позже train
src/pose_av/       метрики, Waymo IO, протоколы eval
tests/             регрессия без скачивания датасета
runs/              логи прогонов (не в git)
weights/           чекпоинты (не в git)
HISTORY.md         что уже мерили
```

Долгие картинки Waymo в git не кладём. Ссылки и команды скачивания — в README ГП1/ГП2. DVC на 387 GB `camera_image` сознательно не вешаем на ГП3.

## Тесты

```powershell
python -m pytest -q
```

Они не требуют GPU, YOLO и parquet-картинок.
