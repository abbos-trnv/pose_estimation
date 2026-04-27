"""
Конвертирует COCO keypoints JSON в YOLO-pose формат.
Все файлы в одной директории.
"""
import os
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

IMAGE_DIRS = [
    'data/images/v1.0-trainval01_keyframes/samples',
    'data/images/v1.0-trainval02_keyframes/samples',
    'data/images/v1.0-trainval03_keyframes/samples',
]

def find_image(filename):
    rel = filename.replace('samples/', '')
    for d in IMAGE_DIRS:
        path = os.path.join(d, rel)
        if os.path.exists(path):
            return path
    return None

def convert_coco_to_yolo(
    keypoints_path='data/annotations/keypoints.json',
    output_dir='data/yolo_pose/images',
):
    """Конвертирует COCO JSON в YOLO-pose .txt формат."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(keypoints_path) as f:
        kp_data = json.load(f)
    
    converted = set()
    for f in os.listdir(output_dir):
        if f.endswith('.txt'):
            converted.add(f.replace('.txt', '.jpg'))
    
    total_imgs = 0
    total_peds = 0
    errors = 0
    skipped = 0
    
    for filename, peds in kp_data.items():
        if filename.replace('samples/', '') in converted:
            skipped += 1
            continue
            
        img_path = find_image(filename)
        if img_path is None:
            errors += 1
            continue
        
        img_path = Path(img_path)
        output_path = Path(output_dir) / (img_path.stem + '.txt')
        
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except:
            errors += 1
            continue
        
        with open(output_path, 'w') as f_out:
            for ped in peds:
                bbox = ped['bbox']
                keypoints = ped['keypoints']
                
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                
                cx_norm = cx / img_w
                cy_norm = cy / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                kp_line = []
                for kp in keypoints:
                    x, y, conf = kp
                    vis = 2 if conf > 0.3 else 1 if conf > 0.1 else 0
                    kp_line.append(f"{x/img_w:.6f} {y/img_h:.6f} {vis}")
                
                line = f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(kp_line)
                f_out.write(line + '\n')
                total_peds += 1
        
        total_imgs += 1
    
    print(f"Конвертировано: {total_imgs} изображений, {total_peds} пешеходов")
    print(f"Уже было: {skipped}")
    print(f"Ошибок (не найдены): {errors}")
    print(f"Сохраено в: {output_dir}")

if __name__ == '__main__':
    convert_coco_to_yolo()