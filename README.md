# pose-av

2D ключевые точки пешеходов по бортовой камере (Waymo Perception).  
Курс *The Missing Semester of Your ML Education*, НИУ ВШЭ, 2026.  
Автор: Тургунов Аббос.

Отчёты: [ГП1](reports/gp1.md) · [ГП2](reports/gp2.md) · [ГП3](reports/gp3.md) · [ГП4](reports/gp4.md)

## Окружение

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -e .
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Облачные логи и DVC (по желанию): `pip install -r requirements-log.txt`.

Eval на картинках: `pip install ultralytics torch`.

### Docker (без GPU, проверка кода)

```powershell
docker compose run --rm tests
docker compose run --rm lint
```

или `.\scripts\run_docker.ps1`.

CI на GitHub делает то же: ruff + pytest.

## Данные

Полный Waymo в git не кладётся. Срез для baseline зафиксирован DVC:

```powershell
pip install dvc
dvc pull    # нужен файл data/kaggle_gp2_slice.zip в кэше или рядом
```

Сегмент якоря: `10023947602400723454_1120_000_1140_000`, 40 кадров. Подробнее: `data/README.md`.

```powershell
$env:WAYMO_ROOT = "data/waymo_v2"
$env:PYTHONPATH = "src"
```

## Оценка и обучение (Hydra)

Конфиги: `configs/` (`configs/README.md`).

```powershell
python scripts/eval_pose.py
python scripts/eval_pose.py model=yolov8n_pose protocol=gt_crop data.max_frames=5
python scripts/export_yolo_pose.py
python scripts/train_pose.py train.epochs=20 train.amp=true
```

Сетка: `python scripts/train_pose.py -m model=yolov8n_pose,yolov8s_pose`.

Артефакты: `runs/<время>/` (метрики + копия конфига). На Kaggle диск сессии временный — те же цифры уходят в W&B, проект **pose-av** (`WANDB_API_KEY`). Без ключа скрипт не падает.

## Структура

```text
configs/     Hydra: данные, модель, eval/train
src/pose_av/ загрузка Waymo, метрики, YOLO, логи, seed
scripts/     CLI
tests/       без GPU и без jpeg
notebooks/   прогон на Kaggle
reports/     вехи
.dvc/        указатели на срез данных
```

## Тесты

```powershell
python -m pytest -q
python -m ruff check src tests scripts
```
