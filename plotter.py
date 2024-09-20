import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import imgviz
from pathlib import Path

'''
background:1, building:2, road:3, water:4, barren:5,forest:6, agriculture:7
background:white, building:red, road:yellow, water:blue, barren:grey, forest:green, agriculture:brown
'''
flat_palette = np.array([
        (0, 0, 0),       # 类别0:黑色
        (255, 255, 255), # 类别1:白色
        (255, 0, 0),     # 类别2:红色
        (255, 255, 0),   # 类别3:黄色
        (0, 0, 255),     # 类别4:蓝色
        (128, 128, 128), # 类别5:灰色
        (0, 255, 0),     # 类别6:绿色
        (165, 42, 42)    # 类别7:棕
        # 可以继续添加更多颜色...
    ], dtype=np.uint8).flatten()

def plt_masked_img(img_path:Path, mask_path):
    img_name = img_path.stem
    print(img_name)
    img_pil = Image.fromarray(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取

    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    lbl_pil.putpalette(flat_palette)

    # 转换 img2_pil 为 RGBA 模式（确保有透明度通道）
    lbl_pil = lbl_pil.convert("RGBA")

    # 获取图像的像素数据
    datas = lbl_pil.getdata()

    # 设置透明度（在这里我们设置为 128，范围是 0-255）
    new_data = []
    for item in datas:
        # item 是一个 (R, G, B, A) tuple
        new_data.append((item[0], item[1], item[2], 128))  # 设置透明度为 128

    # 更新 img2_pil 的数据
    lbl_pil.putdata(new_data)
    
    img_pil.paste(lbl_pil, (0, 0), lbl_pil)
    img_pil.save(f"./output/tmp/{img_name}.png")

def plt_mask_RGB(save_path, from_file:bool, mask_obj=None, mask_path=None):
    if from_file:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        mask = mask_obj
    mask_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colorborad = imgviz.label_colormap()
    mask_pil.putpalette(flat_palette)
    mask_pil.save(save_path)

def plt_mask_batch_RGB(save_path, masks:list):
    width, height = masks[0].shape
    combined_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    for i, mask in enumerate(masks):
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="P").convert("RGBA")
        mask_pil.putpalette(flat_palette)
        combined_image.paste(mask_pil, (0, 0), mask_pil)
    combined_image.save(save_path)


if __name__ == '__main__':
    # plt_masked_img(img_path=Path("dataset/images/0.png"), mask_path="dataset/masks/0.png")
    plt_mask_RGB(from_file=True, mask_path="dataset/labels_png/train/4.png", save_path="output/RGB_mask.png")