"""
Обучение YOLOv8n-pose в Colab.

Инструкция:
1. Создать новый notebook в Colab
2. Скопировать этот код в ячейки
3. Запустить все ячейки
"""

# === Ячейка 1: Установка ===
!pip install ultralytics -q

# === Ячейка 2: Загрузка ключей nuScenes ===
# Вариант А: Скачать с nuScenes сайта (нужен токен)
# 1. Зарегистрироваться на https://www.nuscenes.org/get-data
# 2. Скопировать токен
# 3. Вставить ниже

NUSCENES_TOKEN = "ВАШ_ТОКЕН_ЗДЕСЬ"  # <-- вставить токен

# === Ячейка 3: Скачать данные (только keyframes) ===
import os
import subprocess

# Скачать минимум 3 архива с keyframes
urls = [
    "https://daemon.nuscenes.org/data/v1.0-trainval01_keyframes.tgz",
    "https://daemon.nuscenes.org/data/v1.0-trainval02_keyframes.tgz", 
    "https://daemon.nuscenes.org/data/v1.0-trainval03_keyframes.tgz",
]

# Или использовать wget с токеном
for url in urls:
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        print(f"Скачиваем {filename}...")
        subprocess.run(f"wget -O {filename} --user=USERNAME --password=PASSWORD {url}", shell=True)
        subprocess.run(f"tar -xzf {filename}", shell=True)

# === Ячейка 4: Подготовка папок ===
!mkdir -p data/yolo_pose/images data/yolo_pose/splits

# === Ячейка 5: Конвертация (нужна запустить один раз) ===
# Сначала нужно скопировать скрипт конвертации
# Или использовать готовые файлы если есть

# === Ячейка 6: Обучение ===
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

results = model.train(
    data='/content/data/yolo_pose/nuscenes_pose.yaml',
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    optimizer='AdamW',
    lr0=1e-3,
    pose=12.0,
    project='runs/pose',
    name='yolov8n_nuscenes',
    save=True,
)

# === Ячейка 7: Оценка ===
metrics = model.val()
print(f"Pose mAP50: {metrics.pose.map50:.3f}")
print(f"Pose mAP50-95: {metrics.pose.map:.3f}")

# === Ячейка 8: Скачать модель ===
from google.colab import files
files.download('runs/pose/yolov8n_nuscenes/weights/best.pt')