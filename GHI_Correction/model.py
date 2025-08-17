import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class MultiResNetModel(nn.Module):
    def __init__(self,c_in=1,d_model=512, isimg=False):
        super(MultiResNetModel, self).__init__()
        self.padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=self.padding, padding_mode='zeros', bias=False)
        self.tokenMLP = nn.Sequential(
                    nn.Linear(1, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_model),
                    nn.ReLU()
                )
        self.tokenMLPimg = nn.Sequential(
                    nn.Linear(1, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU()
                )

        # 加载三个预训练的 ResNet18 模型
        self.resnet1 = models.resnet18(pretrained=True)
        self.resnet2 = models.resnet18(pretrained=True)
        self.resnet3 = models.resnet18(pretrained=True)
        self.isimg = isimg

        # 冻结卷积层参数，只训练全连接层
        for model in [self.resnet1, self.resnet2, self.resnet3]:
            # 修改最后的全连接层为特征提取器，输出 256 维特征
            num_features = model.fc.in_features  # ResNet 默认的全连接层输入特征数
            model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(512,256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )
    
         # 定义三个可学习的参数，表示每个图像的贡献
        self.img_weight1 = nn.Parameter(torch.ones(1))  # 高云图像贡献
        self.img_weight2 = nn.Parameter(torch.ones(1))  # 中云图像贡献
        self.img_weight3 = nn.Parameter(torch.ones(1)) # 低云图像贡献
       
        # 合并特征后再通过一个全连接层输出最终结果
        self.fc = nn.Sequential(
            nn.Linear(256*3 +d_model,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_noimg = nn.Sequential(
            nn.Linear(d_model,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x,n):
        n = self.tokenMLP(n.unsqueeze(1))

        if self.isimg ==True:
            # 分别通过三个 ResNet 网络提取特征
            feat1 = self.resnet1(x[:, 0, :, :, :])  # 云层特征
            feat2 = self.resnet2(x[:, 1, :, :, :])  # 中云层特征
            feat3 = self.resnet3(x[:, 2, :, :, :])  # 低云层特征
            _,_,_,h,w=x.shape
             # 根据可学习的权重调整每个图像的贡献
            feat1 = feat1 * self.img_weight1
            feat2 = feat2 * self.img_weight2
            feat3 = feat3 * self.img_weight3
            #print(f"low:{self.img_weight3}")
            # 拼接特征
            combined_features = torch.cat([feat1, feat2, feat3,n], dim=1)  # 在特征维度拼接
            #self.weightlog(self.fc[0].weight)
            # 输出最终结果
            output = self.fc(combined_features)
        else:
            output = self.fc_noimg(n)

        return output

    def weightlog(self,weights):
        # 获取前三个 256 维度部分，具体来说就是权重矩阵的前 3*256 列
        input_dim = 256 * 3  # 输入维度
        selected_weights = weights[:, :input_dim]  # 提取前 3*256 维度的权重，形状 (512, 256*3)

        # 将 selected_weights 按照每个 256 的部分切分
        part_1 = selected_weights[:, :256]  # 对应第一部分 256
        part_2 = selected_weights[:, 256:512]  # 对应第二部分 256
        part_3 = selected_weights[:, 512:768]  # 对应第三部分 256

        # 计算每个部分的模（范数）
        norm_part_1 = torch.norm(part_1)  # 计算每一行的范数，得到形状 (512,)
        norm_part_2 = torch.norm(part_2)  # 同理
        norm_part_3 = torch.norm(part_3)  # 同理

        # 打印每个部分的范数
        print("Norm of Part 1 (256):", norm_part_1)
        print("Norm of Part 2 (256):", norm_part_2)
        print("Norm of Part 3 (256):", norm_part_3)


class MultiResNetModel_1img(nn.Module):
    def __init__(self,c_in=1,d_model=512, isimg=False):
        super(MultiResNetModel_1img, self).__init__()  # 使用正确的 super() 语法
        self.padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=self.padding, padding_mode='zeros', bias=False)
        self.tokenMLP = nn.Sequential(
                    nn.Linear(1, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_model),
                    nn.ReLU()
                )
        self.tokenMLPimg = nn.Sequential(
                    nn.Linear(1, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU()
                )

        # 加载三个预训练的 ResNet18 模型
        self.resnet = models.resnet18(pretrained=True)
        self.isimg = isimg
        # 修改最后的全连接层为特征提取器，输出 256 维特征
        num_features = self.resnet.fc.in_features  # ResNet 默认的全连接层输入特征数
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        

        # 合并特征后再通过一个全连接层输出最终结果
        self.fc = nn.Sequential(
            nn.Linear(256*3 +d_model,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x,n):
        n = self.tokenMLP(n.unsqueeze(1))

        if self.isimg ==True:
            # 分别通过三个 ResNet 网络提取特征
            feat1 = self.resnet(x[:, 0, :, :, :])  # 云层特征
            _,_,_,h,w=x.shape
            # 拼接特征
            mask1=torch.zeros_like(feat1)
            mask2=torch.zeros_like(feat1)
            combined_features = torch.cat([feat1,mask1,mask2,n], dim=1)  # 在特征维度拼接

            # 输出最终结果
            output = self.fc(combined_features)
        else:
            output = self.fc_noimg(n)

        return output

class MultiResNetModel__nwp(nn.Module):
    def __init__(self,c_in=1,d_model=512, isimg=False):
        super(MultiResNetModel__nwp, self).__init__()  # 使用正确的 super() 语法
        self.padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=self.padding, padding_mode='zeros', bias=False)
        self.tokenMLP = nn.Sequential(
                    nn.Linear(1, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_model),
                    nn.ReLU()
                )
        self.tokenMLPimg = nn.Sequential(
                    nn.Linear(1, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU()
                )

        # 加载三个预训练的 ResNet18 模型
        self.resnet = models.resnet18(pretrained=True)
        self.isimg = isimg
        # 修改最后的全连接层为特征提取器，输出 256 维特征
        num_features = self.resnet.fc.in_features  # ResNet 默认的全连接层输入特征数
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        

        # 合并特征后再通过一个全连接层输出最终结果
        self.fc = nn.Sequential(
            nn.Linear(256*3 +d_model,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x,n):
        n = self.tokenMLP(n.unsqueeze(1))

        if self.isimg ==True:
            # 分别通过三个 ResNet 网络提取特征
            feat1 = self.resnet(x[:, 0, :, :, :])  # 云层特征
            _,_,_,h,w=x.shape
            # 拼接特征
            mask1=torch.zeros_like(feat1)
            mask2=torch.zeros_like(feat1)
            mask3=torch.zeros_like(feat1)
            combined_features = torch.cat([mask3,mask1,mask2,n], dim=1)  # 在特征维度拼接

            # 输出最终结果
            output = self.fc(combined_features)
        else:
            output = self.fc_noimg(n)

        return output

