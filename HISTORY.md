# HISTORY

Краткая история измеренных прогонов (ГП2+) и инфры (ГП3). Полные JSON — в `reports/` и `runs/`.

## 2026-09-23 — GP2 baseline (Kaggle CUDA)

- **Код:** `notebooks/gp2_baseline.ipynb` (тот же протокол, что `scripts/eval_pose.py`)
- **Модель:** `yolov8s-pose.pt`, COCO, без fine-tune
- **Данные:** train сегмент `10023947602400723454_1120_000_1140_000`, 40 кадров
- **Конфиг:** `imgsz=1280`, `conf=0.35`, `iou_match=0.3`

| Протокол | matched | unmatched GT | mean OKS | PCK@0.2 | latency |
|---|---|---|---|---|---|
| full_frame | 53 | 100 | 0.956 | 0.464 | 33.8 ms (~29.5 FPS) |
| gt_crop | 150 | 3 | 0.905 | 0.435 | — |

Вывод: top-down по GT-кропу закрывает почти всех пешеходов; bottom-up быстрый, но теряет дальних. Ведущие колонки дальше — PCK@0.2 и покрытие, не одно mean OKS.

Артефакт: `reports/gp2_metrics.json`, overlay `reports/figures/gp2_gt_vs_yolo.jpg`.

## 2026-09-23 — GP3 infra

- Hydra-конфиги `configs/eval.yaml` + `data/` + `model/`
- Скрипт `scripts/eval_pose.py` пишет `runs/<timestamp>/` и шлёт метрики в **W&B** (`WANDB_API_KEY`), опционально MLflow
- Модули `src/pose_av/{data_waymo,evaluate,yolo_backend,runlog,seed,boxes}.py`
- Тесты без датасета: `pytest -q`

## 2026-09-24 — GP4 experiments (deadline)

- E0 COCO 40 frames: crop PCK 0.435 — якорь
- E2 FT 100ep@1280, val 22 frames: crop PCK 0.35; small 0.22. Не лучше COCO.
- Скачаны 6 extra сегментов image+box; scene-split в `gp4_segments.yaml`. Обучение на них — после дедлайна.
