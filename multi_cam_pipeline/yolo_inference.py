import cv2
from ultralytics import YOLO

# Load a model
pretrain_weights_path = "/home/leedh/바탕화면/MCT_for_ChamDog/model_training/runs/train/weights/best.pt"
model = YOLO(pretrain_weights_path)  # load a custom model

sample_image1 = "/home/leedh/바탕화면/MCT_for_ChamDog/model_training/dataset/valid/images/V006_77_0_00_10_01_13_0_b08_20201020_0003_S01_1.jpg"
sample_image2 = "/home/leedh/바탕화면/MCT_for_ChamDog/model_training/dataset/valid/images/V006_77_0_00_10_01_13_0_b08_20201020_0008_S01_1.jpg"

im1 = cv2.imread(sample_image1)
result1 = model.predict(source=im1, save=True, save_txt=True)
print(result1)

results = model.predict(source=[sample_image1, sample_image2], save=True, save_txt=True)