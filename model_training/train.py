from ultralytics import YOLO

# Load a model
model = YOLO("pretrain_weights/yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="dataset/data.yaml", epochs=200, imgsz=640, patience=20, workers=8)