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

---

## Датасет: nuScenes

### Скачано
- `v1.0-trainval01_keyframes.tgz` (4.5 GB)
- `v1.0-trainval02_keyframes.tgz` (4.3 GB)
- `v1.0-trainval03_keyframes.tgz` (4.2 GB)
- `v1.0-trainval_meta.tgz` (440 MB)

### Статистика (full trainval)
| Метрика | Значение |
|---------|----------|
| Уникальных семплов | 34,149 |
| Всего keyframes | 409,788 |
| CAM_FRONT | 102,447 |
| Всего аннотаций | 1,166,187 |
| Пешеходов (instances) | 11,512 |

### Формат данных
- **Изображения**: `samples/CAM_FRONT/*.jpg` (и прочие камеры)
- **Аннотации**: 3D bbox в мировых координатах (translation, size, rotation + LiDAR points)
- **Нет keypoints** — только bbox

### Категории пешеходов
- `human.pedestrian.adult` — взрослые
- `human.pedestrian.child` — дети
- `human.pedestrian.wheelchair` — инвалидные кресла
- `human.pedestrian.stroller` — коляски
- `human.pedestrian.personal_mobility` — самокаты, сигвеи
- `human.pedestrian.police_officer` — полицейские
- `human.pedestrian.construction_worker` — строители

### Структура метаданных
```
v1.0-trainval/
├── sample_annotation.json  # 3D bbox
├── sample_data.json         # изображения
├── sample.json            # семплы
├── instance.json          # объекты
├── category.json        # категории
├── calibrated_sensor.json  # калибровка камер
├── ego_pose.json        # поза авто
├── visibility.json    # видимость
└── ...
```

---

## План работ

1. [ ] Изучить данные (визуально)
2. [ ] Фильтрация кропов (размер, occlusion)
3. [ ] Выбрать teacher модель по бенчмаркам
4. [ ] Teacher → псевдоразметка
5. [ ] Student → обучение

---

## Заметки

### 2026-04-26
- Структура репо создана
- 3/9 keyframes скачано
- blobs_camera перенесён в trash/
- GitHub: https://github.com/abbos-trnv/pose_estimation