import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import imgviz
from pathlib import Path

def plt_masked_img(img_path:Path, mask_path):
    img_name = img_path.stem
    print(img_name)
    img_pil = Image.fromarray(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取

    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())

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

def plt_mask_RGB(mask_path, save_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取

    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


if __name__ == '__main__':
    # plt_masked_img(img_path=Path("dataset/images/0.png"), mask_path="dataset/masks/0.png")
    plt_mask_RGB(mask_path="output/test_2.png", save_path="output/RGB_mask_2.png")