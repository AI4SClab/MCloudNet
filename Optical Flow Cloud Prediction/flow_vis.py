

import cv2
import numpy as np
import os
#from skimage.metrics import structural_similarity as ssim
from datetime import datetime
def center_top_crop(img, target_height, target_width):
    """裁剪图像到指定的高度和宽度，采用中心顶部裁剪方式"""
    h, w, _ = img.shape
    top = 0  # 从顶部开始裁剪
    left = (w - target_width) // 2  # 水平居中裁剪
    cropped_img = img[top:top + target_height, left:left + target_width]
    return cropped_img

def bottom_right_crop(img, target_height, target_width):
    """裁剪图像到指定的高度和宽度，采用右下角裁剪方式"""
    h, w, _ = img.shape  # 获取图像的高度、宽度和通道数
    bottom = h  # 右下角裁剪，下边界为图像的底部
    right = w  # 右下角裁剪，右边界为图像的右侧
    top = bottom - target_height  # 计算上边界
    left = right - target_width  # 计算左边界
    cropped_img = img[top:bottom, left:right]  # 裁剪图像
    return cropped_img
# 遍历指定目录下的文件
def top_left_crop(img, target_height, target_width):
    """裁剪图像到指定的高度和宽度，采用左上角裁剪方式"""
    h, w, _ = img.shape  # 获取图像的高度、宽度和通道数
    top = 0  # 左上角裁剪，上边界为图像的顶部
    left = 0  # 左上角裁剪，左边界为图像的左侧
    bottom = top + target_height  # 计算下边界
    right = left + target_width  # 计算右边界
    cropped_img = img[top:bottom, left:right]  # 裁剪图像
    return cropped_img
directory = '/root/cloud/cloudflower/vitdata/mid/'

def center_crop(img, target_height, target_width):
    """
    裁剪图像到指定的高度和宽度，采用中心裁剪方式
    :param img: 输入图像（numpy数组）
    :param target_height: 目标高度
    :param target_width: 目标宽度
    :return: 裁剪后的图像
    """
    h, w, _ = img.shape  # 获取图像的高度、宽度和通道数
    # 计算中心点坐标
    center_x, center_y = w // 2, h // 2
    # 计算裁剪区域的边界
    left = center_x - target_width // 2
    right = center_x + target_width // 2
    top = center_y - target_height // 2
    bottom = center_y + target_height // 2
    # 裁剪图像
    cropped_img = img[top:bottom, left:right]
    return cropped_img
mse_total1 = 0
ssim_total1 = 0
mse_total3 = 0
ssim_total3 = 0
count = 0

start_dt = datetime.strptime("20190111_0610", "%Y%m%d_%H%M")
end_dt = datetime.strptime("20190111_0650", "%Y%m%d_%H%M")

# 遍历目录并筛选文件
all_files = []
for root, dirs, files in os.walk(directory):
    dirs.sort()  # 按字母顺序排序子目录
    files.sort()  # 按字母顺序排序文件
    for file in files:
        try:
            # 提取时间部分
            date_part = file.split('_')[1] + '_' + file.split('_')[2].split('.')[0]
            # 将时间部分转换为 datetime 对象
            dt = datetime.strptime(date_part, "%Y%m%d_%H%M")

            # 检查文件时间是否在指定范围内
            if start_dt <= dt <= end_dt:
                all_files.append(file)
        except (IndexError, ValueError):
            # 忽略不符合日期格式的文件
            continue

# 按三张连续的方式读取
for i in range(len(all_files) - 2):
    print(len(all_files))
    file1 = all_files[i]
    file2 = all_files[i + 1]
    file3 = all_files[i + 2]

    # 读取图像
    frame1 = cv2.imread(os.path.join(directory, file1))
    frame2 = cv2.imread(os.path.join(directory, file2))
    frame3 = cv2.imread(os.path.join(directory, file3))

    h1, w1, _ = frame1.shape
    h2, w2, _ = frame2.shape
    h3, w3, _ = frame3.shape

    if h1!=h2 or h2!=h3 or h3!=h1:
        # 找到最小的高度和宽度
        # min_height = min(h1, h2, h3)
        # min_width = min(w1, w2, w3)
        min_height = 80
        min_width =80

        # 按中心顶部裁剪到最小尺寸
        # frame1 = center_top_crop(frame1, min_height, min_width)
        # frame2 = center_top_crop(frame2, min_height, min_width)
        # frame3 = center_top_crop(frame3, min_height, min_width)

    
    # 检查图像是否成功读取
    if frame1 is None or frame2 is None:
        print(f'Failed to read {file1} or {file2}')
        continue
    if frame3 is None:
        break

    # 转换为灰度图
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 计算光流
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    
    # 获取光流场
    flow_x = flow[..., 0]  # 水平光流
    flow_y = flow[..., 1]  # 垂直光流
    
   

    scale = 8  # 光流缩放因子，用于延长箭头
    thickness = 1  # 箭头线条粗细
    color = (0, 255, 0)  # 箭头颜色（绿色）
    step = 16  # 步长

    h, w = prev.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            # 获取每个点的光流
            fx, fy = flow[y, x]
            # 放大光流以延长箭头
            end_x = int(x + fx * scale)
            end_y = int(y + fy * scale)
            # 绘制箭头
            cv2.arrowedLine(frame1, (x, y), (end_x, end_y), color, thickness, tipLength=0.2)


   
    cv2.imwrite(f'/root/cloud/cloudflower/flow/arr_{file1}', frame1)


