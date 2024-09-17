from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo/yolov8n-seg.yaml').load('yolo/yolov8n-seg.pt')  # Load model and weights
    model.train(data='./dataset/dataset.yaml', epochs=100)  # Train for 3 epochs

