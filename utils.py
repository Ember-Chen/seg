import base64
from openai import OpenAI
from ultralytics import YOLO
import numpy as np
from plotter import plt_masked_img, plt_mask_RGB, plt_mask_batch_RGB
import requests
from pathlib import Path

KEY = 'sk-f2a66cEo8vYFMq8ssj91rQFzFoldfUqtbRWCJJ3l2LYgnF07'
client = OpenAI(api_key=KEY, base_url="https://api.f2gpt.com/v1")


model = YOLO('yolo/yolov8n-seg.yaml').load('yolo/best.pt')
TOTAL_PIXELS = 640*640
CLASS_DICT = {'1': 'Background', '2': 'Building', '3': 'Road', '4': 'Water', '5': 'Barren', '6': 'Forest', '7': 'Agriculture'}
OLLAMA_URL = 'http://localhost:6006/api/generate'
PROMPT_SEG = f'''
This is an image segmentation map where each color represents a specific land cover type. The image consists of large, distinct color blocks. Below is the color mapping:

- **Blue**: Water bodies
- **Green**: Forests
- **Red**: Buildings
- **Brown**: Agriculture
- **White**: Background

Please describe the distribution of these land cover types within the image.
For example, you might describe an image where "The water bodies are concentrated in the top-left corner. The forests occupies the bottom-right. The buildings are scattered along the top edge, and the agriculture area dominate the center of the image."
'''
PROMPT_NO_SEG = f'''
    Please analyze distribution of Water bodies, Forests, Buildings, Agriculture and Barren land in the image.
'''
PROMPT_PIE_CHART = f'''
    Please generate pie chart with mermaid code, using GIVEN_DATA. \n
    Title is fixed: Distribution\n
    mermaid code example:
    ```mermaid
    pie title 123 
        "A":0.4
        "B":0.6
    ``` \n
    only return the mermaid code.
'''
PROMPT_XY_CHART = f'''
    Please generate XY chart with mermaid code, using GIVEN_DATA. \n
    Title is fixed: Distribution\n
    mermaid code example:
    ```mermaid
    xychart-beta
    title "Sales Revenue"
        x-axis [jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec]
        y-axis "Revenue (in $)" 4000 --> 11000
        bar [5000, 6000, 7500, 8200, 9500, 10500, 11000, 10200, 9200, 8500, 7000, 6000]
    ``` \n
    only return the mermaid code.
'''


def predict(img_path:Path):
    '''返回图片保存路径，返回占比dict'''
    # 修改透明度：ultralytics/ultralytics/utils/plotting.py
    results = model.predict(
        img_path,
        save=True, conf=0.35, show_labels=True, show_conf=True, show_boxes=True
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

def get_descriptions(img_path):
    img_b64 = img_to_base64(img_path)
    res = get_img_description(img_b64)
    return res


def gen_chart(data, chart_type:str):
    prompt = PROMPT_PIE_CHART if chart_type == 'pie' else PROMPT_XY_CHART
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
            ]}
        ],
        temperature=0.0,
    )
    print(response.choices[0].message.content)


def get_img_description(img_path:Path, prompt:str):
    base64_image = img_to_base64(img_path)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )

    print(response.choices[0].message.content)

if __name__ == '__main__':
    # res = predict(Path(r'dataset/images/train/5.png'))
    # print('\n')
    # print(res)
    # pass
    # gen()
    get_img_description(Path(r'dataset/images/train/4.png'), PROMPT_NO_SEG)
