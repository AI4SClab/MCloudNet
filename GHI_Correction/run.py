
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
from dataloader import ImageDataset,ImageDataset_1img
from model import MultiResNetModel
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_contribution(epochs,losses,img_weights1,img_weights2,img_weights3,current_time):
    with PdfPages(f"/root/cloud/data/resnet18_res/{current_time}+weight.pdf") as pdf:
        # 绘制损失曲线
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(16, 9))
        plt.plot(epochs, losses, label='Loss', color='red', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.grid(True)
        plt.legend()
        pdf.savefig()  # 保存到 PDF
        plt.close()

        # 绘制图片权重曲线
        # plt.figure(figsize=(16, 6))
        # plt.plot(epochs, img_weights1, label='High cloud weight', color='blue', marker='o')
        # plt.plot(epochs, img_weights2, label='Mid cloud weight', color='green', marker='o')
        # plt.plot(epochs, img_weights3, label='Low cloud weight', color='orange', marker='o')
        # plt.xlabel('Epoch')
        # plt.ylabel('Weight')
        # plt.title('Image Weights over Epochs')
        # plt.grid(True)
        # plt.legend()
        # 设置字体大小和线条宽度
        plt.figure(figsize=(16, 6))
        plt.plot(epochs, img_weights1, label='High cloud weight', color='blue', marker='s', linewidth=4, markersize=12)  # 正方形标记
        plt.plot(epochs, img_weights2, label='Mid cloud weight', color='green', marker='^', linewidth=4, markersize=12)  # 上三角标记
        plt.plot(epochs, img_weights3, label='Low cloud weight', color='orange', marker='D', linewidth=4, markersize=12)  # 菱形标记

        # 设置字体大小
        plt.xlabel('Epoch', fontsize=22)
        plt.ylabel('Weight', fontsize=22)
        plt.title('Image Weights over Epochs', fontsize=22)

        # 设置图例字体大小
        plt.legend(fontsize=20)

        # 设置网格
        plt.grid(True)
        pdf.savefig()  # 保存到 PDF
        plt.close()

# 配置日志文件，确保以追加模式写入日志
logging.basicConfig(
    filename="/root/cloud/result/resent/nwpresnet.log",  # 日志文件名
    level=logging.INFO,       # 日志记录级别
    # format="%(asctime)s - %(message)s",  # 日志格式，包含时间戳
    filemode="a"              # 追加模式
)
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 选择此次用的训练变量
fix_seed = 2023
set_seed(fix_seed)
num_epochs = 15
# num_epochs = 1 # 暂时为了其他数据
isimg=True
imgsize =16
std1 = 0.229
#logging.info(f"std1:{std1}")
std2 = 0.220
#logging.info(f"std2:{std2}")
std3 = 0.225
#logging.info(f"std3:{std3}")
# 数据预处理
transform = transforms.Compose([
    transforms.CenterCrop((imgsize)),  # 调整到预训练模型的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[std1,std2,std3]),
])
# 数据集和数据加载器
data = True # 是：ImageDataset 否：消融实验
logging.info(f"data:{data}")
if data is True:
    img_dir = "/root/cloud/cloudflower/flow4resnet/Dual/"
    csv_file = "/root/cloud/data/mydata/station07_pred.csv"
    #csv_file = "/root/cloud/data/mydata/itrans.csv"
    dataset = ImageDataset(img_dir, csv_file, transform=transform)

else:
    img_dir = "/root/cloud/cloudflower/vitdata/cloud_image/"
    csv_file = "/root/cloud/data/mydata/station04_pred.csv"
    dataset = ImageDataset_1img(img_dir, csv_file, transform=transform)

# 定义学习率调整函数
def sincos_learning_rate(epoch, base_lr=0.01, a=0.5, b=0.1, c=0.5, d=0.1):
    """
    动态计算学习率
    :param epoch: 当前训练的 epoch
    :param base_lr: 基础学习率
    :param a, b, c, d: sincos 函数的系数
    :return: 动态学习率
    """
    return base_lr * (a * math.sin(b * epoch) + c * math.cos(d * epoch))

# 定义训练集和测试集的大小
train_size = int(0.8 * len(dataset))  # 80% 用作训练集
print(f"train size:{train_size}")

test_size = len(dataset) - train_size  # 剩下的用作测试集
print(f"test size:{test_size}")

# 随机划分数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
print("ok")
# 加载预训练模型
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
model = MultiResNetModel(isimg=isimg).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



# 训练模型

# 初始化早停参数
patience = 3  # 容忍验证损失未改善的最大次数
min_delta = 1e-4  # 验证损失改善的最小值
best_val_loss = float('inf')  # 初始化最优验证损失
patience_counter = 0  # 连续未改善次数计数器


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

saved_data = []
epochs = []
losses = []
img_weights1 = []
img_weights2 = []
img_weights3 = []

for epoch in range(num_epochs):
    # 动态调整学习率
    lr = sincos_learning_rate(epoch, base_lr=0.001, a=0.5, b=0.1, c=0.5, d=0.1)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 更新优化器的学习率
    # 打印当前学习率
    print(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {lr:.6f}")
    if epoch ==0 :
        logging.info(f"base Learning Rate: {lr:.6f}")
        logging.info(f"batch:{train_loader.batch_size}")

    running_loss = 0.0
    
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for (images,nwp_ghi,variables), label in train_iterator:
        images = images.to(device)  # 图像张量
        nwp = nwp_ghi
        nwp = nwp.to(device)  # 数值特征
        label = label.to(device)  # 标签
        # print("NWP shape:", nwp.shape)
        # print("Label shape:", label.shape)
        # 前向传播
        outputs = model(images, nwp).float().squeeze(1)
        loss = criterion(outputs, label)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 更新 tqdm 显示的损失
        train_iterator.set_postfix(loss=running_loss / (train_iterator.n + 1))  # 显示平均损失

        
        # 提取 datetime 和 outputs
        datetime_info = variables.get('datetime', None)  # 假设 datetime 存储在 variables 字典中
        outputs = dataset.denormalize_outputs(outputs)
        
        # 将 datetime 和 outputs 转换为可以保存的格式
        if epoch==num_epochs-1 and datetime_info is not None:
            for dt, output in zip(datetime_info, outputs):
                saved_data.append([dt, output.item()])  # 保存为列表，output.item() 转为纯数值

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    # 打印图片权重
    print(f"High cloud image weight (img_weight1): {model.img_weight1.item():.4f}")
    print(f"Mid cloud image weight (img_weight2): {model.img_weight2.item():.4f}")
    print(f"Low cloud image weight (img_weight3): {model.img_weight3.item():.4f}")
    # 记录每个 epoch 的数值
    epochs.append(epoch + 1)
    losses.append(running_loss / len(train_loader))
    img_weights1.append(model.img_weight1.item())
    img_weights2.append(model.img_weight2.item())
    img_weights3.append(model.img_weight3.item())

# 保存训练数据到 CSV 文件
train_file =f"/root/cloud/data/resnet18_res/{current_time}+train.csv"
with open(train_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['date_time', 'Output_ghi'])  # 写入表头
    writer.writerows(saved_data)  # 写入数据
print(f"Data saved to {train_file}")
save_contribution(epochs,losses,img_weights1,img_weights2,img_weights3,current_time)


# 测试模式下禁用梯度计算
model.eval()
all_outputs = []
all_labels = []
all_nwp = []
all_date = []
total_test_loss =0
total_nwp_loss =0
loss_real_nwp =0
loss_real_test =0
with torch.no_grad():
    for (images,nwp_ghi,variables), label in test_loader:
        images = images.to(device)  # 图像张量
        nwp=nwp_ghi
        nwp = nwp.to(device)  # 数值特征
        label = label.to(device)  # 标签

        # 前向传播
        outputs = model(images,nwp).float().squeeze(1)
        loss = criterion(outputs, label)  # 计算测试集的损失

        loss_nwp = criterion(nwp,label)
        total_test_loss += loss.item()  # 累积损失
        total_nwp_loss += loss_nwp.item()
        # batch_mymse = np.mean((labels.numpy() - outputs.detach().numpy()) ** 2)
        # total_test_loss += batch_mymse
        # batch_mse = np.mean((labels.numpy() - nwp.detach().numpy()) ** 2)
        # total_nwp_loss += batch_mse
        # 反归一化
        outputs = dataset.denormalize_outputs(outputs)
        label = dataset.denormalize_outputs(label)
        nwp = dataset.denormalize_outputs(nwp)
        loss_real_test += criterion(outputs,label)
        loss_real_nwp += criterion(nwp,label)
        # 保存预测值和标签
        all_outputs.extend(outputs.cpu().numpy())  # 转为 NumPy 数组，便于保存
        all_labels.extend(label.cpu().numpy())
        all_nwp.extend(nwp.cpu().numpy())
        all_date.extend(variables['datetime'])

        
print(f"The number of outputs in all_outputs is: {len(all_outputs)}")
# 打印测试集的平均损失
average_test_loss = total_test_loss / len(test_loader)
average_nwp_loss = total_nwp_loss / len(test_loader)
average_real_loss_test = loss_real_test /len(test_loader)
average_real_nwp_loss = loss_real_nwp / len(test_loader)
print(f"Test Loss: {average_test_loss:.6f}")
print(f"nwp Loss: {average_nwp_loss:.6f}")
print(f"Test real Loss: {average_real_loss_test:.6f}")
print(f"nwp real Loss: {average_real_nwp_loss:.6f}")


logging.info(f"begin!seed:{fix_seed}")
logging.info(f"csv_file:{csv_file}")
logging.info(f"isimg:{isimg}")
logging.info(f"num_epochs:{num_epochs}")
logging.info(f"imgsize:{imgsize}")
logging.info(f"train size:{train_size}")
logging.info(f"test size:{test_size}")
logging.info(f"current_time:{current_time}")
logging.info(f"Test Loss: {average_test_loss:.6f}")
logging.info(f"Tnwp Loss: {average_nwp_loss:.6f}")
logging.info(f"Test real Loss: {average_real_loss_test:.6f}")
logging.info(f"nwp real Loss: {average_real_nwp_loss:.6f}")
# 将预测值和标签保存到 CSV 文件

output_csv_path = f"/root/cloud/data/resnet18_res/{current_time}+test.csv"
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["date_time","nwp_ghi","Output_ghi", "Label_lmdghi"])  # 写入表头
    writer.writerows(zip(all_date,all_nwp,all_outputs, all_labels))  # 写入数据行

logging.info(f"Testing results saved to {output_csv_path}")
logging.info("end!!!")
# # 保存模型
# torch.save(model.state_dict(), f"/root/cloud/data/resnet18model/{current_time}.pth")
# print("Training complete and model saved.")


def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只打印需要训练的参数
            print(f"Parameter name: {name}")
            print(f"Shape: {param.shape}")
            print(f"Value: {param.data}\n")

#print_model_parameters(model)

def img_contribute(model):
    # 查看 self.fc 层的权重
    fc_weights = model.fc[0].weight.data  # 获取 fc 层第一层线性层的权重（shape: [1, 256*3 + d_model]）

    # 对应特征的权重
    weights_img = fc_weights[0, :256*3].reshape(3, 256)  # 对应三张图的权重，按每张图 256 特征进行分割
    weights_nwp = fc_weights[0, 256*3:]  # 数值天气预报特征的权重
    # 计算每张图像的贡献
    img_contributions = weights_img.sum(dim=1)  # 对每张图像的 256 特征加总，得到每张图像的贡献

    # 输出每张图像的贡献值
    print("High cloud contribution:", img_contributions[0].item())
    print("Mid cloud contribution:", img_contributions[1].item())
    print("Low cloud contribution:", img_contributions[2].item())

torch.save(model.state_dict(), 'ourmodel.pth')