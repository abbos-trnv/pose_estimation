# Training on Cloud (Kaggle / Google Colab)

## Quick Start

### Option 1: Google Colab (Recommended)

1. Upload project to Google Drive:
   ```
   Pose_estimation/
   ├── data/yolo_pose/          # ~17021 .txt files + splits
   └── notebooks/colab_training.ipynb
   ```

2. Open `colab_training.ipynb` in Colab

3. Run cells in order

### Option 2: Kaggle

1. Create new Kaggle Notebook

2. Upload files via:
   - Kaggle API (`kaggle datasets download`)
   - Or upload directly to notebook

3. Use `kaggle_training.ipynb` as reference

## Data Needed

The training data is in `data/yolo_pose/`:
- `images/` - 17021 .txt files (YOLO pose format)
- `splits/` - train.txt, val.txt, test.txt
- `nuscenes_pose.yaml` - dataset config

## GPU Time

- ~15-30 minutes for 50 epochs (T4/P100 GPU)
- ~1-2 hours for 100 epochs

## Expected Results

- pose mAP50-95: ~0.4-0.6 (depends on data quality)
- Latency: <10ms on GPU, <50ms on CPU

## After Training

1. Download `best.pt` from notebook
2. Save to `weights/` folder
3. Test locally:
   ```python
   from ultralytics import YOLO
   model = YOLO('weights/best.pt')
   model('test_image.jpg')
   ```