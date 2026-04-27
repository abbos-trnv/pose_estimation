"""
Train YOLOv8n-pose на nuScenes данных.
"""
from ultralytics import YOLO
import torch

def train_yolo_pose():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Стартовая модель
    model = YOLO('weights/yolov8n-pose.pt')
    
    # Обучение
    results = model.train(
        data='data/yolo_pose/nuscenes_pose.yaml',
        epochs=50,          # Для baseline - 50 эпох
        imgsz=640,
        batch=16,          # На MPS нельзя большой batch
        device=device,
        optimizer='AdamW',
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        pose=12.0,
        kobj=2.0,
        project='runs/pose',
        name='yolov8n_nuscenes_v1',
        save=True,
        save_period=10,
        val=True,
        plots=True,
        patience=20,
        workers=4,
    )
    
    print("Training complete!")
    return results

if __name__ == '__main__':
    train_yolo_pose()