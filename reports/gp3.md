# Детекция ключевых точек пешеходов в сценах автономного вождения

**Отчёт по вехе 3 (ГП3): инфраструктура**

Тургунов Аббос  
Курс: Дополнительные главы машинного обучения / *The Missing Semester of Your ML Education*  
НИУ ВШЭ, 23 сентября 2026 г.  
Репозиторий: `pose_estimation/`

---

## 1. Что оценивает веха

По курсу: **архитектура репозитория + качество кода (включая README)** и **воспроизводимость экспериментов + логирование артефактов**.

Это не новая сетка моделей и не fine-tune. Baseline ГП2 остаётся цифрой в `HISTORY.md`; здесь его можно **повторить скриптом** с тем же протоколом.

Лекция «Инфраструктура ML-проектов»: эксперименты в скриптах, параметры в Hydra/OmegaConf, логи рядом с конфигом, README чтобы не спрашивали «как запустить».

---

## 2. Архитектура репозитория

```text
pose_estimation/
├── configs/
│   ├── eval.yaml              # вход Hydra
│   ├── data/waymo_slice.yaml
│   └── model/yolov8s_pose.yaml, yolov8n_pose.yaml
├── src/pose_av/
│   ├── keypoints.py, metrics.py   # ГП1
│   ├── boxes.py, data_waymo.py    # матчинг и parquet
│   ├── evaluate.py                # full_frame / gt_crop
│   ├── yolo_backend.py            # Ultralytics (опционально)
│   ├── seed.py, runlog.py
│   └── catalog.py
├── scripts/eval_pose.py
├── tests/
├── notebooks/gp2_baseline.ipynb   # архив измеренного ГП2
├── reports/
├── runs/                          # артефакты прогонов, не в git
├── README.md
├── HISTORY.md
├── Makefile
├── pyproject.toml
└── requirements.txt
```

Ноутбуки больше не источник правды для eval: ячейки ГП2 перенесены в `evaluate.py` + `YoloPoseBackend`. Конфиг не хардкодится в Python.

Сознательно **нет** DVC на полный `camera_image` (~387 GB) и нет Docker на этом шаге: соло-проект, данные уже качаются точечно через gcloud, тяжёлый прогон — Kaggle. Путь к срезу задаётся `WAYMO_ROOT` / `data.waymo_root`.

---

## 3. Воспроизводимость

- **Seed:** `pose_av.seed.set_seed` (python / numpy / torch).
- **Конфиг:** Hydra 1.3. Любой ключ с CLI, без правки файлов.
- **Зависимости:** `requirements.txt` + `pip install -e .`. YOLO/torch — отдельно, чтобы `pytest` жил без GPU.
- **Команда, эквивалентная ГП2:**

```text
python scripts/eval_pose.py model=yolov8s_pose data.max_frames=40 protocol=both
```

(`imgsz=1280`, `conf=0.35`, сегмент как в отчёте ГП2.)

---

## 4. Логирование артефактов

Каждый запуск Hydra создаёт `runs/<timestamp>/` (локальный слепок: `metrics.json`, `config.yaml`). На Kaggle эта папка умрёт вместе с сессией, поэтому те же метрики уходят **в облако**:

- **Weights & Biases** — включён по умолчанию. Нужен `WANDB_API_KEY` (локально `wandb login`, на Kaggle: Add-ons → Secrets). Проект `pose-av`. Без ключа eval не падает.
- **MLflow** — выключен, пока не задан `logging.mlflow.enabled=true` и `tracking_uri` (Dagshub или свой сервер). Лекция также упоминает ClearML; W&B выбран как самый простой путь с Kaggle GPU.

Код: `src/pose_av/runlog.py` (`write_run` + `log_remote`).


---

## 5. Качество кода

- Пакет `src/pose_av`, тесты через `pythonpath = src`.
- Unit-тесты метрик + IoU + фейковый backend eval **без** скачивания Waymo.
- Ruff (`make lint`). Makefile: `install`, `test`, `eval`.
- README: быстрый старт, eval, дерево, что не кладём в git.

---

## 6. Что дальше (ГП4)

Сетка уже крутится оверрайдами Hydra: `model=yolov8n_pose/yolov8s_pose`, `imgsz`, `conf`, `protocol`, список сегментов. Fine-tune и teacher ViTPose — отдельные конфиги `model/`, не новые тетрадки.
