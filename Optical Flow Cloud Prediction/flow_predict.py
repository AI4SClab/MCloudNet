import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
def center_top_crop(img, target_height, target_width):
    """裁剪图像到指定的高度和宽度，采用中心顶部裁剪方式"""
    h, w, _ = img.shape
    top = 0  # 从顶部开始裁剪
    left = (w - target_width) // 2  # 水平居中裁剪
    cropped_img = img[top:top + target_height, left:left + target_width]
    return cropped_img
# 目标保存路径
output_dir = r'/root/cloud/cloudflower/flow4resnet/Dual02/low/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历指定目录下的文件

directory = '/root/cloud/cloudflower/vitdata02/low/'


mse_total1 = 0
ssim_total1 = 0
mse_total3 = 0
ssim_total3 = 0
count = 0

start_dt = datetime.strptime("20180701_0000", "%Y%m%d_%H%M")
end_dt = datetime.strptime("20190612_0000", "%Y%m%d_%H%M")

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
        min_height = min(h1, h2, h3)
        min_width = min(w1, w2, w3)

        # 按中心顶部裁剪到最小尺寸
        frame1 = center_top_crop(frame1, min_height, min_width)
        frame2 = center_top_crop(frame2, min_height, min_width)
        frame3 = center_top_crop(frame3, min_height, min_width)
    
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
    
    # 根据光流预测下一帧
    h, w = prev.shape
    next_frame = np.zeros_like(frame1)  # 初始化预测的下一帧
    
    # 构建一个网格来表示每个像素的坐标
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 根据光流更新像素的坐标
    next_x = grid_x + flow_x
    next_y = grid_y + flow_y
    
    # 将坐标限制在图像范围内（避免溢出）
    next_x = np.clip(next_x, 0, w - 1).astype(np.float32)
    next_y = np.clip(next_y, 0, h - 1).astype(np.float32)
    
    # 使用光流场映射得到下一帧
    next_frame = cv2.remap(frame2, next_x, next_y, interpolation=cv2.INTER_LINEAR)
    # 保存预测的下一帧
    output_filename = os.path.join(output_dir, f'predicted_{file3}')  # 保存到指定路径
    suc = cv2.imwrite(output_filename, next_frame)
    
    if suc:
        print(f'Saved {output_filename}')
    else:
        print(f'Failed to save {output_filename}')

   
