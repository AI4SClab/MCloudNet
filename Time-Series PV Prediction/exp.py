from date_loader import PowerPredictionDataset
from model import MLPnew
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import LambdaLR
import math
import random
import numpy as np
import csv

# 设置 CSV 文件路径
train_file = '/root/cloud/data/4itrans/2025-01-18_12-34-10+train.csv'  # 替换为你的 CSV 文件路径
test_file = '/root/cloud/data/4itrans/2025-01-18_12-34-10+test.csv'
output_file = '/root/cloud/data/finalres/2025-01-18_12-34-10.csv'  # 输出文件路径（可以与输入文件相同）

# 修改后的代码
def run_experiment_with_random_seed(seed, train_file, test_file, output_file, num_epochs=5):
    # 设置随机种子
    random.seed(seed)           
    np.random.seed(seed)        
    torch.manual_seed(seed)     
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  

    # 确保 cuDNN 的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建数据加载器
    train_dataset = PowerPredictionDataset(train_file)
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    test_dataset = PowerPredictionDataset(test_file)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

    # 初始化 MLP 模型
    input_size = 2  # 输入特征数量，'itr_output' 和 'Output_ghi'
    hidden_size = 10  # 隐藏层神经元数量
    output_size = 1  # 输出为一个数值（power）
    num_layers = 3
    dropout_rate = 0.1
    output_token_dim = 64
    fix_ghi_token_dim = 16
    model = MLPnew(input_size, hidden_size, output_size, num_layers, dropout_rate, 
                output_token_dim, fix_ghi_token_dim)

    # 使用 GPU（如果可用）
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    model = model.to(device) 

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失

    # 训练模型
    train_model(train_loader, model, criterion, device, num_epochs)

    # 测试并保存结果
    output_file_with_seed = f"{output_file.replace('.csv', '')}_seed{seed}.csv"
    mse,mae = test_model(test_loader, model, device, output_file_with_seed)

    # 返回随机种子下的评估结果
    return seed, mse,mae,output_file_with_seed

def sincos_learning_rate(epoch, base_lr=0.01, a=0.5, b=0.1, c=0.5, d=0.1):
    """
    动态计算学习率
    :param epoch: 当前训练的 epoch
    :param base_lr: 基础学习率
    :param a, b, c, d: sincos 函数的系数
    :return: 动态学习率
    """
    min_lr = 0
    lr = base_lr * (a * math.sin(b * epoch) + c * math.cos(d * epoch))
    return max(lr, min_lr)  # 设置一个最低的学习率阈值


# 训练模型
def train_model(train_loader, model, criterion,  device, num_epochs=5):
    model.train()  # 设置模型为训练模式
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scheduler = LambdaLR(optimizer, lr_lambda)
    for epoch in range(num_epochs):
        running_loss = 0.0
        # lr = sincos_learning_rate(epoch, base_lr=0.005, a=0.5, b=0.1, c=0.5, d=0.1)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr  # 更新优化器的学习率
            # 打印当前学习率
        # print(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {lr:.6f}")

        for i, (inputs, labels) in enumerate(train_loader):
            # 将标签和输入放到 GPU 上（如果可用）
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs.flatten(), labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            #scheduler.step()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# 测试模型并保存预测结果
def test_model(test_loader, model, device, output_file):
    model.eval()  # 设置模型为评估模式
    predictions = []
    true_values = []
    pvafteritrans = []
    fixedghi = []
    # 遍历 DataLoader，进行预测
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
            
            outputs = model(inputs)
            # Move tensors to CPU and convert to numpy arrays
            outputs_cpu = outputs.cpu().numpy()
            labels_cpu = labels.cpu().numpy()  # Convert labels to CPU numpy

            # Inverse normalization
            outputs, true_labels = test_dataset.inverse_normalize(
                outputs_cpu, labels_cpu
            )

            # Extend predictions and true values
            predictions.extend(outputs)
            true_values.extend(true_labels)  # Flatten labels if needed
        
    # Ensure both true_values and predictions are numpy arrays
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    # 计算评估指标
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    # # 将结果保存到一个新的 DataFrame
    # results_df = pd.DataFrame({
    #     'true_values': true_values,
    #     'predicted_power': predictions,
    #     'pv_after_itrans': pvafteritrans,
    #     'fixed_ghi': fixedghi
    # })
    # # 将新结果保存到 CSV 文件
    # results_df.to_csv(output_file, index=False)
    # print(f"Predictions and metrics saved to {output_file}")
    return mse,mae


# 创建数据加载器                                            
train_dataset = PowerPredictionDataset(train_file)
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
test_dataset = PowerPredictionDataset(test_file)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)


# print(f"input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}, "
#       f"num_layers={num_layers}, dropout_rate={dropout_rate}, "
#       f"output_token_dim={output_token_dim}, fix_ghi_token_dim={fix_ghi_token_dim}", 
#       f"epoch={num_epochs}",end="")
def save_results_to_csv(results, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['Random Seed', 'MSE', 'MAE'])
        
        # 写入每个实验的结果
        for result in results:
            seed, mse, mae, result_file = result
            writer.writerow([seed, mse, mae,])
# 进行多个实验，记录所有种子的结果
results = []
for seed in range(1,10):
    seed,mse, mae, result_file = run_experiment_with_random_seed(
        seed, train_file, test_file, output_file, num_epochs=200)
    results.append((seed, mse, mae, result_file))
    # run_experiment_with_random_seed(seed, train_file, test_file, output_file, num_epochs=100)


# 打印所有结果
for result in results:
    print(f"Random Seed: {result[0]} | Results saved in: {result[1]} {result[2]}")

# save_results_to_csv(results, '/root/cloud/data/finalres/results_with_seeds1-20_lr1.csv')
