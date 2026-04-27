from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection

print("Скачиваю детектор людей (RT-DETR)...")
det_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
det_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
print("Детектор готов")

print("Скачиваю ViTPose-base...")
pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
print("ViTPose готов")

print("Всё скачано успешно!")
