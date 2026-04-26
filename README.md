# Pose Estimation

Детекция ключевых точек пешеходов в сценах автономного вождения.

## Подход

1. Teacher модель (YOLOv8-pose) → псевдоразметка на кропах пешеходов
2. Student модель → обучение на псевдоразметке
3. Сжатие (квантизация, дистилляция, прунинг) → target latency ~100мс

## Стек

- PyTorch
- Ultralytics (YOLOv8)
- DVC (версионирование данных)
- Hydra (конфигурации)

## Структура

```
.dvc/          - файлы DVC
configs/       - конфиги Hydra
notebooks/     - EDA ноутбуки
src/           - исходный код
tests/         - тесты
weights/       - сохранённые веса
trash/         - ненужные файлы
```

## Датасет

nuScenes (keyframes only)