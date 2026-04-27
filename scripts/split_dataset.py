"""
Разделяет данные на train/val/test по сценам nuScenes.
"""
import os
import json
import random
from pathlib import Path

def split_by_scenes(
    keypoints_path='data/annotations/keypoints.json',
    output_dir='data/yolo_pose/splits',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """Разделяет по scene_token, не случайно."""
    
    random.seed(seed)
    
    with open(keypoints_path) as f:
        kp_data = json.load(f)
    
    # Группируем по scene (из имени файла)
    # Формат: samples/CAM_FRONT/xxx.jpg -> n015-2018-07-18-11-07-57+0800
    scenes = {}
    for filename in kp_data.keys():
        # Извлекаем token из имени
        parts = filename.split('/')
        if len(parts) >= 2:
            # n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530xxx.jpg
            name = parts[-1].replace('.jpg', '')
            # n015-2018-07-18-11-07-57+0800
            scene_token = name.split('__')[0]
            if scene_token not in scenes:
                scenes[scene_token] = []
            scenes[scene_token].append(filename)
    
    scene_list = list(scenes.keys())
    random.shuffle(scene_list)
    
    n_train = int(len(scene_list) * train_ratio)
    n_val = int(len(scene_list) * val_ratio)
    
    train_scenes = scene_list[:n_train]
    val_scenes = scene_list[n_train:n_train + n_val]
    test_scenes = scene_list[n_train + n_val:]
    
    print(f"Всего сцен: {len(scene_list)}")
    print(f"Train: {len(train_scenes)} сцен")
    print(f"Val: {len(val_scenes)} сцен")
    print(f"Test: {len(test_scenes)} сцен")
    
    # Собираем файлы
    train_files = []
    val_files = []
    test_files = []
    
    for scene, files in scenes.items():
        if scene in train_scenes:
            train_files.extend(files)
        elif scene in val_scenes:
            val_files.extend(files)
        else:
            test_files.extend(files)
    
    print(f"\nИзображений:")
    print(f"  Train: {len(train_files)}")
    print(f"  Val: {len(val_files)}")
    print(f"  Test: {len(test_files)}")
    
    # Сохраняем
    os.makedirs(output_dir, exist_ok=True)
    
    for name, files in [('train.txt', train_files), ('val.txt', val_files), ('test.txt', test_files)]:
        filepath = os.path.join(output_dir, name)
        with open(filepath, 'w') as f:
            for file in files:
                # Найдем полный путь к изображению
                rel = file.replace('samples/', '')
                for d in ['data/images/v1.0-trainval01_keyframes/samples',
                        'data/images/v1.0-trainval02_keyframes/samples',
                        'data/images/v1.0-trainval03_keyframes/samples']:
                    full_path = os.path.join(d, rel)
                    if os.path.exists(full_path):
                        f.write(full_path + '\n')
                        break
    
    print(f"\nСохранено в: {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoints', default='data/annotations/keypoints.json')
    parser.add_argument('--output', default='data/yolo_pose/splits')
    args = parser.parse_args()
    
    split_by_scenes(args.keypoints, args.output)