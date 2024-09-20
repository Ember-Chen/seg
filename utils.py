import base64
from ultralytics import YOLO
import numpy as np
from plotter import plt_masked_img, plt_mask_RGB, plt_mask_batch_RGB
import requests
from pathlib import Path

model = YOLO('yolo/yolov8n-seg.yaml').load('yolo/best.pt')
TOTAL_PIXELS = 640*640
CLASS_DICT = {'1': 'Background', '2': 'Building', '3': 'Road', '4': 'Water', '5': 'Barren', '6': 'Forest', '7': 'Agriculture'}
OLLAMA_URL = 'http://localhost:6006/api/generate'
PROMPT = f'''
This is an image segmentation map where each color represents a specific land cover type. The image consists of large, distinct color blocks. Below is the color mapping:

- **Blue**: Water bodies
- **Green**: Forests
- **Red**: Buildings
- **Brown**: Agriculture
- **White**: Background

Please describe the distribution of these land cover types within the image.
For example, you might describe an image where "the blue area, representing water bodies, is concentrated in the top-left corner, while the green area, representing forests, occupies the bottom-right. The red areas representing buildings are scattered along the top edge, and the brown areas for agriculture dominate the center of the image."
'''

def predict(img_path:Path):
    '''返回图片保存路径，返回占比dict'''
    # 修改透明度：ultralytics/ultralytics/utils/plotting.py
    results = model.predict(
        img_path,
        save=True, conf=0.35, show_labels=False, show_conf=False, show_boxes=False
    )
    result = results[0]  # Get the first result
    masks = result.masks
    masks = result.masks.data  # 获取掩码数据
    classes = result.boxes.cls.int().tolist()  # 获取每个掩码对应的类别，并转为整数列表
    pixel_counts = {name: 0 for name in result.names.values()}
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
    pixel_percentages = {
        CLASS_DICT[category]: (count / TOTAL_PIXELS)
        for category, count in pixel_counts.items()
        if count > 0
    }
    img_save_path = result.save_dir + f'\\{img_path.name}'
    return img_save_path, pixel_percentages


# 读取图片并转换为base64格式
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# 发送图片数据给OLLAMA
def invoke(img_b64):
    # 构造请求体
    payload = {
        'model': 'llava:34b',
        'prompt': PROMPT,
        'stream': False,
        'images': [img_b64],
        "options": {
            "temperature": 0
        }
    }

    # 发送POST请求
    response = requests.post(OLLAMA_URL, json=payload)

    # 打印返回的结果
    if response.status_code == 200:
        res = response.json()['response']
        print(f"Response: {res}")
        return res
    else:
        print(f"Failed with status code: {response.status_code}")

def get_descriptions(img_path):
    img_b64 = img_to_base64(img_path)
    res = invoke(img_b64)
    return res

if __name__ == '__main__':
    res = predict(Path(r'dataset/images/train/0.png'))
    print('\n')
    print(res)

