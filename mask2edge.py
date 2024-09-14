import cv2
import numpy as np


mask = cv2.imread('dataset/masks/0.png', cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

height, width = mask.shape
normalized_contours = []

for class_value in range(0, 8):
    # 创建当前类别的二值mask
    class_mask = np.uint8(mask == class_value)
    
    # 检查当前类别的mask是否存在目标
    if cv2.countNonZero(class_mask) == 0:
        continue  # 如果没有目标，跳过

    # 寻找当前类别的轮廓
    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        contour = contour.reshape(-1, 2)  # 转换为Nx2的坐标格式
        normalized_contour = contour / [width, height]  # 归一化坐标到0~1

        normalized_contours.append({
            'class_id': class_value,  # 类别ID从0开始
            'contour': normalized_contour.tolist()
        })

with open('output/path_to_label.txt', 'w') as f:
    for item in normalized_contours:
        class_id = item['class_id']
        contour = np.array(item['contour'])

        # 构建标签字符串
        label_str = str(class_id)
        
        # 添加多边形点坐标
        for point in contour:
            label_str += f" {point[0]} {point[1]}"
        
        f.write(label_str + '\n')