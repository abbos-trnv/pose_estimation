import json
from pathlib import Path

p = Path(r"D:\MissingML\pose_estimation\notebooks\gp2_baseline.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

infer = r'''from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_NAME)
print("device", device)

def gt_vec(rec):
    xy = np.zeros((14, 2), dtype=np.float64)
    vis = np.zeros((14,), dtype=np.float64)
    tmap = {int(t): j for j, t in enumerate(rec["types"])}
    for k, t in enumerate(CAMERA_ORDER):
        if t not in tmap:
            continue
        j = tmap[t]
        xy[k] = [rec["x"][j], rec["y"][j]]
        vis[k] = 1
    return xy, vis

def run_full_frame(im):
    t0 = time.perf_counter()
    pred = model.predict(im, verbose=False, device=device, conf=CONF, imgsz=IMGSZ, classes=[0])[0]
    dt = time.perf_counter() - t0
    p_boxes, p_kpts = [], []
    if pred.boxes is not None and pred.keypoints is not None and len(pred.boxes):
        xyxy = pred.boxes.xyxy.cpu().numpy()
        kxy = pred.keypoints.xy.cpu().numpy()
        kconf = pred.keypoints.conf.cpu().numpy() if pred.keypoints.conf is not None else np.ones(kxy.shape[:2])
        for bi in range(len(xyxy)):
            p_boxes.append(xyxy[bi])
            vis = (kconf[bi] > 0.2).astype(np.float64)
            p_kpts.append(coco17_to_waymo14(kxy[bi], vis))
    return pred, p_boxes, p_kpts, dt

def run_on_crop(im, xyxy):
    h, w = im.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    pad_x, pad_y = int(0.2 * (x2 - x1)), int(0.2 * (y2 - y1))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None
    crop = im[y1:y2, x1:x2]
    pred = model.predict(crop, verbose=False, device=device, conf=0.1, imgsz=256, classes=[0])[0]
    if pred.keypoints is None or len(pred.keypoints) == 0:
        return None
    kxy = pred.keypoints.xy.cpu().numpy()[0].copy()
    kconf = pred.keypoints.conf.cpu().numpy()[0] if pred.keypoints.conf is not None else np.ones(17)
    kxy[:, 0] += x1
    kxy[:, 1] += y1
    vis = (kconf > 0.15).astype(np.float64)
    return coco17_to_waymo14(kxy, vis)

oks_ff, pck_ff, miss_ff, lat = [], [], 0, []
oks_td, pck_td, miss_td = [], [], 0
viz_candidates = []

for fi, key in enumerate(keys):
    jpeg = imgs[key]
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        continue
    pred, p_boxes, p_kpts, dt = run_full_frame(im)
    lat.append(dt)
    box_by_oid = {b["oid"]: b for b in boxes.get(key, [])}
    frame_gt = []
    max_area = 0.0
    for rec in hkps[key]:
        b = box_by_oid.get(rec["oid"])
        if b is None:
            miss_ff += 1
            miss_td += 1
            continue
        gt_xyxy = [b["cx"]-b["w"]/2, b["cy"]-b["h"]/2, b["cx"]+b["w"]/2, b["cy"]+b["h"]/2]
        gt_xy, gt_vis = gt_vec(rec)
        if gt_vis.sum() == 0:
            continue
        max_area = max(max_area, b["w"] * b["h"])
        frame_gt.append((gt_xyxy, gt_xy, gt_vis, b))
        best_i, best_iou = -1, IOU_MATCH
        for i, pb in enumerate(p_boxes):
            val = iou_xyxy(gt_xyxy, pb)
            if val > best_iou:
                best_iou, best_i = val, i
        if best_i < 0:
            miss_ff += 1
        else:
            pr_xy, _ = p_kpts[best_i]
            wh = np.array([b["w"], b["h"]])
            oks_ff.append(mean_oks(gt_xy, pr_xy, gt_vis, wh))
            pck_ff.append(pck(gt_xy, pr_xy, gt_vis, wh, 0.2))
        crop_pred = run_on_crop(im, gt_xyxy)
        if crop_pred is None:
            miss_td += 1
        else:
            pr_xy, _ = crop_pred
            wh = np.array([b["w"], b["h"]])
            oks_td.append(mean_oks(gt_xy, pr_xy, gt_vis, wh))
            pck_td.append(pck(gt_xy, pr_xy, gt_vis, wh, 0.2))
    viz_candidates.append((max_area, im[:, :, ::-1].copy(), frame_gt, p_boxes, p_kpts))
    if (fi + 1) % 10 == 0:
        print(f"{fi+1}/{len(keys)}  ff_matched={len(oks_ff)} td_matched={len(oks_td)}")

viz_candidates.sort(key=lambda x: -x[0])
viz = viz_candidates[:4]

def pack(name, oks, pck, miss, extra=None):
    d = {
        "protocol": name,
        "matched_instances": len(oks),
        "unmatched_gt": miss,
        "mean_OKS": float(np.nanmean(oks)) if oks else None,
        "PCK@0.2": float(np.nanmean(pck)) if pck else None,
    }
    if extra:
        d.update(extra)
    return d

summary = {
    "model": MODEL_NAME,
    "device": device,
    "imgsz": IMGSZ,
    "conf": CONF,
    "segment": SEGMENT,
    "frames": len(keys),
    "full_frame_bottom_up": pack("full_frame", oks_ff, pck_ff, miss_ff, {
        "latency_ms_mean": float(np.mean(lat)*1000) if lat else None,
        "fps": float(1.0/np.mean(lat)) if lat else None,
    }),
    "gt_crop_top_down": pack("gt_crop", oks_td, pck_td, miss_td),
}
print(json.dumps(summary, indent=2))
'''

viz = r'''import matplotlib.pyplot as plt

SKELETON = [(0,13),(1,2),(1,3),(3,5),(2,4),(4,6),(1,7),(2,8),(7,8),(7,9),(9,11),(8,10),(10,12)]

def draw_skel(ax, xy, vis, color, lw=1.5):
    for a, b in SKELETON:
        if vis[a] and vis[b]:
            ax.plot([xy[a,0], xy[b,0]], [xy[a,1], xy[b,1]], color=color, lw=lw)
    m = vis > 0
    if m.any():
        ax.scatter(xy[m,0], xy[m,1], s=12, c=color)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, item in zip(axes.ravel(), viz):
    _, rgb, frame_gt, p_boxes, p_kpts = item
    ax.imshow(rgb)
    ax.axis("off")
    for gt_xyxy, gt_xy, gt_vis, b in frame_gt:
        x1, y1, x2, y2 = gt_xyxy
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec="red", lw=1.2))
        draw_skel(ax, gt_xy, gt_vis, "red")
    for pb, (pr_xy, pr_vis) in zip(p_boxes, p_kpts):
        ax.add_patch(plt.Rectangle((pb[0], pb[1]), pb[2]-pb[0], pb[3]-pb[1], fill=False, ec="lime", lw=1.0, ls="--"))
        draw_skel(ax, pr_xy, pr_vis, "lime")
plt.suptitle("red = Waymo GT  |  lime = YOLO full-frame  |  кадры с самыми крупными GT-пешеходами")
plt.tight_layout()
plt.show()
'''

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if src.startswith("from ultralytics import YOLO"):
        cell["source"] = [ln + "\n" for ln in infer.splitlines()]
    if src.startswith("import matplotlib.pyplot"):
        cell["source"] = [ln + "\n" for ln in viz.splitlines()]

cfg_src = "".join(nb["cells"][1]["source"])
cfg_src = cfg_src.replace(
    'MAX_FRAMES = 40          # поставь None чтобы прогнать весь сегмент\nIOU_MATCH = 0.3\nMODEL_NAME = "yolov8n-pose.pt"\n',
    'MAX_FRAMES = 40          # None = весь сегмент\nIOU_MATCH = 0.3\nCONF = 0.35\nIMGSZ = 1280             # мелкие пешеходы, 640 мало\nMODEL_NAME = "yolov8s-pose.pt"  # n слишком слаб на дальних людях\n',
)
nb["cells"][1]["source"] = [ln + "\n" for ln in cfg_src.splitlines()]

p.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("ok")
