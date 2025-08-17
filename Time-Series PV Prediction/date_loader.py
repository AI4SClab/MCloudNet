import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
class PowerPredictionDataset(Dataset):
    def __init__(self, csv_file, seqlen=1):
        """
        初始化数据集，读取 CSV 文件并提取所需列
        """
        self.data = pd.read_csv(csv_file)

        # 滑动窗口的长度
        self.seqlen = seqlen

        # 提取输入特征
        self.outputs = self.data['output'].values  # 单列特征 'output'
        self.fix_ghi = self.data['fix_ghi'].values  # 单列特征 'fix_ghi'

        # 提取标签
        self.y = self.data['label'].values

        # 创建 StandardScaler 实例
        self.scaler_outputs = StandardScaler()
        self.scaler_fix_ghi = StandardScaler()
        self.scaler_y = StandardScaler()

        # 对 'fix_ghi' 特征进行归一化
        fix_ghi_original_shape = self.fix_ghi.shape  # 保存原始形状
        self.fix_ghi = self.scaler_fix_ghi.fit_transform(self.fix_ghi.reshape(-1, 1))
        self.fix_ghi = self.fix_ghi.reshape(fix_ghi_original_shape)  # 恢复原始形状

        # 对 'outputs' 特征进行归一化
        outputs_original_shape = self.outputs.shape  # 保存原始形状
        self.outputs = self.scaler_outputs.fit_transform(self.outputs.reshape(-1, 1))
        self.outputs = self.outputs.reshape(outputs_original_shape)  # 恢复原始形状

        # 对 'y' 标签进行归一化
        y_original_shape = self.y.shape  # 保存原始形状
        self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1))
        self.y = self.y.reshape(y_original_shape)  # 恢复原始形状
        
        
    def __len__(self):
        """
        返回数据集的大小
        """
        #return len(self.data)
        return len(self.data) - self.seqlen + 1
    
    def __getitem__(self, idx):
        # """
        # 根据索引返回一条数据（输入和标签）
        # """
        # x = torch.tensor(self.x[idx], dtype=torch.float32)
        # y = torch.tensor(self.y[idx], dtype=torch.float32)
        # return x, y

        # 滑动窗口选取 fix_ghi 特征
        ghi_window = self.fix_ghi[idx:idx + self.seqlen]  # 长度为 seqlen 的滑动窗口

        # 当前样本的 'output' 值
        output = self.outputs[idx + self.seqlen - 1]  # 滑动窗口的最后一个点对应的输出

        # 构造输入 x（包含滑动窗口的 fix_ghi 和对应的 output）
        x = torch.tensor([output] + ghi_window.tolist(), dtype=torch.float32)  # 拼接 output 和 fix_ghi 窗口

        # 当前样本对应的标签
        y = torch.tensor(self.y[idx + self.seqlen - 1], dtype=torch.float32)

        return x, y

    def inverse_normalize(self, outputs,inputs, y):
        """
        反归一化
        """
        # Inverse transform the scaled data
        inputs_original_shape = inputs.shape  # 保存原始形状
        inputs = self.scaler_outputs.inverse_transform(inputs.reshape(-1, 1))
        inputs = inputs.reshape(inputs_original_shape)  # 恢复原始形状

        out_original_shape =outputs.shape  # 保存原始形状
        out = self.scaler_y.inverse_transform(outputs.reshape(-1, 1))
        out = out.reshape(out_original_shape)  # 恢复原始形状

        y_original_shape = y.shape  # 保存原始形状
        y = self.scaler_y.inverse_transform(y.reshape(-1, 1))
        y = y.reshape(y_original_shape)  # 恢复原始形状

        # Return the flattened arrays
        return out,inputs,y
        
