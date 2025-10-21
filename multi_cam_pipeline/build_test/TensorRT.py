from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "weights", "best.pt")
model = YOLO(weights_path)

# Export the model to TensorRT
model.export(format="engine", half=True)