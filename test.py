from ultralytics import YOLO

model = YOLO('yolo/yolov8n-seg.yaml').load('yolo/yolov8n-seg.pt')  # Load model and weights

# model.train(data='data/coco128.yaml', epochs=3)  # Train for 3 epochs


