from ultralytics import YOLO
import yaml

data = {
    'train': '/home/sonujha/rnd/Wheat-Head-Detection-YOLO/data/preprocessed/images/train',
    'val': '/home/sonujha/rnd/Wheat-Head-Detection-YOLO/data/preprocessed/images/valid',
    'nc': 1,
    'names': ['wheat']
}

with open('wheat.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Display model information (optional)
print(model.info())

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="wheat.yaml", epochs=10, imgsz=640)
