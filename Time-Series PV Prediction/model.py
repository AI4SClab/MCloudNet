import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=5, dropout_rate=0.3, 
                 output_token_dim=64, fix_ghi_token_dim=2):
        super(MLP, self).__init__()
        self.output_token_dim = output_token_dim
        self.fix_ghi_token_dim = fix_ghi_token_dim

        # 映射 output 和 fix_ghi 为 token
        self.output_token_layer = nn.Linear(1, output_token_dim)  # output 映射到更高维
        self.fix_ghi_token_layer = nn.Linear(1, fix_ghi_token_dim)  # fix_ghi 映射到较低维
        
        # 主 MLP 模型
        self.layers = nn.ModuleList()  # 用于存储所有层
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # 输入层
        self.layers.append(nn.Linear(output_token_dim + fix_ghi_token_dim, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.BatchNorm1d(hidden_size))
        
        # 中间的隐藏层
        for _ in range(num_layers - 2):  # 中间层数量由 num_layers 控制
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.BatchNorm1d(hidden_size))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        # x 应包含 output 和 fix_ghi 两部分特征
        output, fix_ghi = x[:, 0:1], x[:, 1:2]  # 假设 x 的前两列为 output 和 fix_ghi

        # 将 output 和 fix_ghi 映射为 token
        output_token = self.output_token_layer(output)
        fix_ghi_token = self.fix_ghi_token_layer(fix_ghi)

        # 将 token 拼接为整体输入
        combined_token = torch.cat([output_token, fix_ghi_token], dim=1)

        # 通过 MLP
        for layer in self.layers:
            combined_token = layer(combined_token)
        return combined_token

class MLPnew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout_rate=0.3, 
                 output_token_dim=64, fix_ghi_token_dim=2):
        super(MLPnew, self).__init__()
        self.output_token_dim = output_token_dim
        self.fix_ghi_token_dim = fix_ghi_token_dim
        
        # 主 MLP 模型
        self.layers = nn.ModuleList()  # 用于存储所有层
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # 输入层
        self.layers.append(nn.Linear(input_size, hidden_size))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(dropout_rate))
        # self.layers.append(nn.BatchNorm1d(hidden_size))
        
        # 中间的隐藏层
        for _ in range(num_layers - 2):  # 中间层数量由 num_layers 控制
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            # self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(dropout_rate))
            # self.layers.append(nn.BatchNorm1d(hidden_size))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        # x 应包含 output 和 fix_ghi 两部分特征
        output, fix_ghi = x[:, 0:1], x[:, 1:2]  # 假设 x 的前两列为 output 和 fix_ghi

        # 将 token 拼接为整体输入
        combined_token = torch.cat([output, fix_ghi], dim=1).squeeze(-1)

        # 通过 MLP
        for layer in self.layers:
            combined_token = layer(combined_token)
        return combined_token

