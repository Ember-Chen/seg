import cv2
from ultralytics import YOLO
import numpy as np

# 加载YOLOv8模型
model = YOLO('yolov8n-seg.pt')  # 使用适当的模型路径

# 加载图像
img_path = 'demo_data/terrain2.png'
image = cv2.imread(img_path)

# 使用模型进行预测
results = model.predict(source=img_path, conf=0.35, save=True)  # 降低置信度阈值

# 获取第一个预测结果
result = results[0]

# 检查是否有掩码数据
if result.masks is not None:
    print("Found masks!")
else:
    print("No masks found.")

# 获取图像的高度和宽度
height, width = image.shape[:2]
total_pixels = height * width

# 初始化字典来存储每个类别的像素数量
pixel_counts = {name: 0 for name in result.names.values()}

# 如果有掩码数据，则进行处理
if result.masks is not None:
    masks = result.masks.data  # 获取掩码数据
    classes = result.boxes.cls.int().tolist()  # 获取每个掩码对应的类别，并转为整数列表

    # 遍历每个掩码并统计每个类别的像素数量
    for mask, class_id in zip(masks, classes):
        # 确保mask是一个布尔数组
        binary_mask = mask.to('cpu').numpy() > 0.5
        
        # 检查class_id是否在names字典中
        if int(class_id) in result.names:
            # 使用类别名作为键来增加像素计数
            category_name = result.names[int(class_id)]
            pixel_counts[category_name] += np.sum(binary_mask)

    # 计算每个类别的像素占比
    pixel_percentages = {category: count / total_pixels * 100 for category, count in pixel_counts.items() if count > 0}

    # 打印结果
    for category, percentage in pixel_percentages.items():
        print(f"{category}: {percentage:.2f}%")

# 如果没有找到掩码数据，给出提示
else:
    print("No segmentation masks found in the prediction.")