"""
Train YOLOv8n-pose локально.

Запустить: .venv/bin/python scripts/train_local.py
"""
from ultralytics import YOLO

def train():
    # Используем GPU если доступен
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Модель - автозагрузится при первом запуске
    model = YOLO('yolov8n-pose.pt')
    
    # Обучение
    model.train(
        data='data/yolo_pose/nuscenes_pose.yaml',
        epochs=50,        # baseline: 50 эпох
        imgsz=640,
        batch=16,         # на MPS 16, на GPU 32+
        device=device,
        optimizer='AdamW',
        lr0=1e-3,
        weight_decay=5e-4,
        warmup_epochs=3,
        fliplr=0.5,
        mosaic=1.0,
        pose=12.0,
        kobj=2.0,
        project='runs/pose',
        name='yolov8n_nuscenes',
        save=True,
        save_period=10,
        patience=20,
    )
    
    print("Готово!")
    print("Модель: runs/pose/yolov8n_nuscenes/weights/best.pt")

if __name__ == '__main__':
    train()