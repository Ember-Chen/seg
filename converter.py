import cv2
import numpy as np
from pathlib import Path

def png_2_txt(png_path:Path, txt_save_path:Path):
    mask = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
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

    with open(txt_save_path, 'w') as f:
        for item in normalized_contours:
            contour = np.array(item['contour'])
            label_str = str(item['class_id'])
            
            # 添加多边形点坐标
            for point in contour:
                label_str += f" {point[0]} {point[1]}"
            
            f.write(label_str + '\n')


def txt_2_png(txt_path:Path, png_save_path:Path):
    blank_mask = np.zeros((1024, 1024), dtype=np.uint8)
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            class_id = int(line[0])
            points = np.array(line[1:], dtype=np.float32).reshape(-1, 2)
            points = (points * 1024).astype(np.int32)
            points = points.tolist()
            cv2.drawContours(blank_mask, [np.array(points)], -1, class_id, thickness=-1)
    cv2.imwrite(str(png_save_path), blank_mask)


def png_2_txt_batch(masks_png_path:Path, masks_txt_path:Path):
    for mask_path in masks_png_path.glob('*.png'):
        png_2_txt(
            png_path=mask_path,
            txt_save_path= (masks_txt_path / f"{mask_path.stem}.txt")
        )


if __name__ == '__main__':
    # png_2_txt(mask_path='dataset/masks/0.png')
    # txt_2_png(contours_txt_path=r'output/contours.txt')
    png_2_txt_batch(
        masks_png_path=Path('dataset/masks_png/val'),
        masks_txt_path=Path('dataset/labels/val')
    )
    # txt_2_png(txt_path=Path('dataset/masks_txt/2.txt'), png_save_path=Path('output/test_2.png'))
    pass
