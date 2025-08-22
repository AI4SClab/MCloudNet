
import os
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader ,random_split
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
import math
import random
import numpy as np

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(csv_file)


        self.labels['date_time'] = pd.to_datetime(self.labels['date_time'], errors='coerce')

        self.labels = self.labels[
        (self.labels['date_time'] >= '2018-07-01') &
        (self.labels['date_time'] < '2019-07-02') &
        (self.labels['13high'].notna()) & 
        (self.labels['13mid'].notna()) &
        (self.labels['13low'].notna()) &
        (self.labels['power']!=0)
    ]

        self.transform = transform
        self.global_min = min(self.labels['lmd_totalirrad'].min(), self.labels['nwp_globalirrad'].min())
        self.global_max = max(self.labels['lmd_totalirrad'].max(), self.labels['nwp_globalirrad'].max())

        self.folder_mapping = {
            'high': 'high',
            'mid': 'mid',
            'low': 'low'
        }
        self.image_columns = ['13high', '13mid', '13low']
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        images = []

        # 加载图像
        img_names = [row[col] for col in self.image_columns]
        for col, img_name in zip(['high', 'mid', 'low'], img_names):
            img_path = f"{self.img_dir}{self.folder_mapping[col]}/{img_name}"

            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image)  # 应用图像转换
                images.append(image)
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found at path: {img_path}")

        # 将图像列表转换为张量
        images = torch.stack(images)  # [3, C, H, W], 其中 3 是图像数量

        # 处理数值特征
        label = float(row['lmd_totalirrad'])
        # label = (label - self.label_min) / (self.label_max - self.label_min)  # 标签归一化
        label = (row['lmd_totalirrad'] - self.global_min) / (self.global_max - self.global_min)
        label = torch.tensor(label, dtype=torch.float32)


        nwp_ghi = float(row['nwp_globalirrad'])
        nwp_ghi = (row['nwp_globalirrad'] - self.global_min) / (self.global_max - self.global_min)
        nwp_ghi = torch.tensor(nwp_ghi, dtype=torch.float32)


        current_time = row['date_time'].strftime("%Y-%m-%d_%H-%M-%S") if isinstance(row['date_time'], pd.Timestamp) else str(row['date_time'])
        variables = {
            'datetime':current_time,
            'nwp_ghi': row['nwp_globalirrad'],
            'nwp_dni': row['nwp_directirrad'],
            'lmd_temperature': row['nwp_temperature'],
            'lmd_humidity': row['nwp_humidity'],
            'lmd_windspeed': row['nwp_windspeed'],
            'lmd_winddirection': row['nwp_winddirection'],
            'lmd_pressure': row['nwp_pressure'],
            'lmd_ghi': row['lmd_totalirrad'],
            'lmd_dni': row['lmd_diffuseirrad'],
            'lmd_temperature': row['lmd_temperature'],
            'lmd_pressure': row['lmd_pressure'],
            'lmd_winddirection': row['lmd_winddirection'],
            'lmd_windspeed': row['lmd_windspeed'],
            'power': row['power']
        }  
        # 将字典中的每个值转换为Tensor，并以变量名为键保存
        tensor_variables = {key: (torch.tensor(value, dtype=torch.float32) if key != 'datetime' else value) 
                            for key, value in variables.items()}
        # 返回图像张量和数值特征
        return (images,nwp_ghi,tensor_variables),label

    def denormalize_outputs(self,outputs):
        return outputs * (self.global_max - self.global_min) + self.global_min

# 数据预处理
transform = transforms.Compose([
    transforms.CenterCrop((16)),  # 调整到预训练模型的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x



# 消融实验 只用未分层的云图
class ImageDataset_1img(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(csv_file)

        # 转换 'date_time' 列为 datetime 类型
        self.labels['date_time'] = pd.to_datetime(self.labels['date_time'], errors='coerce')

        # 过滤出 'date_time' 为 2019 年 1 月的数据
        self.labels = self.labels[
        (self.labels['date_time'] >= '2018-07-01') &
        (self.labels['date_time'] < '2019-07-02') &
        (self.labels['13high'].notna()) & # 过滤出 img 列不为空的行
        (self.labels['13mid'].notna()) &
        (self.labels['13low'].notna()) &
        (self.labels['img'].notna()) & # 过滤出 img 列不为空的行
        (self.labels['power']!=0)
    ]

        self.transform = transform
        self.global_min = min(self.labels['lmd_totalirrad'].min(), self.labels['nwp_globalirrad'].min())
        self.global_max = max(self.labels['lmd_totalirrad'].max(), self.labels['nwp_globalirrad'].max())

        # Columns for images and NWP features
        self.image_columns = ['img', 'img', 'img']
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        images = []

        # 加载图像
        img_names = [row[col] for col in self.image_columns]
        for img_name in img_names:
            img_path = f"{self.img_dir}{img_name}"
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image)  # 应用图像转换
                images.append(image)
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found at path: {img_path}")

        # 将图像列表转换为张量
        images = torch.stack(images)  # [3, C, H, W], 其中 3 是图像数量

        # 处理数值特征
        label = float(row['lmd_totalirrad'])
        # label = (label - self.label_min) / (self.label_max - self.label_min)  # 标签归一化
        label = (row['lmd_totalirrad'] - self.global_min) / (self.global_max - self.global_min)
        label = torch.tensor(label, dtype=torch.float32)

     

        nwp_ghi = float(row['nwp_globalirrad'])
        nwp_ghi = (row['nwp_globalirrad'] - self.global_min) / (self.global_max - self.global_min)
        nwp_ghi = torch.tensor(nwp_ghi, dtype=torch.float32)


        current_time = row['date_time'].strftime("%Y-%m-%d_%H-%M-%S") if isinstance(row['date_time'], pd.Timestamp) else str(row['date_time'])
        variables = {
            'datetime':current_time,
            'nwp_ghi': row['nwp_globalirrad'],
            'nwp_dni': row['nwp_directirrad'],
            'lmd_temperature': row['nwp_temperature'],
            'lmd_humidity': row['nwp_humidity'],
            'lmd_windspeed': row['nwp_windspeed'],
            'lmd_winddirection': row['nwp_winddirection'],
            'lmd_pressure': row['nwp_pressure'],
            'lmd_ghi': row['lmd_totalirrad'],
            'lmd_dni': row['lmd_diffuseirrad'],
            'lmd_temperature': row['lmd_temperature'],
            'lmd_pressure': row['lmd_pressure'],
            'lmd_winddirection': row['lmd_winddirection'],
            'lmd_windspeed': row['lmd_windspeed'],
            'power': row['power']
        }  
        # 将字典中的每个值转换为Tensor，并以变量名为键保存
        tensor_variables = {key: (torch.tensor(value, dtype=torch.float32) if key != 'datetime' else value) 
                            for key, value in variables.items()}
        # 返回图像张量和数值特征
        return (images,nwp_ghi,tensor_variables),label

    def denormalize_outputs(self,outputs):
        return outputs * (self.global_max - self.global_min) + self.global_min
