from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo/yolov8n-seg.yaml').load('yolo/best.pt')
    class_names = ['NotSure', 'Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture']
    results = model.predict(r'dataset/images/train/4.png', save=True, conf=0.35) # Run inference on a single image
    result = results[0]  # Get the first result
    print(results)
