from data_provider.data_factory import data_provider

from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import StandardScaler
import csv
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.train_data,self.train_loader,self.vali_data,self.vali_loader,self.test_data,self.test_loader = self._get_data()
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        # 训练集和测试不在同一个文件
        if self.args.isdiff:
            train_set,test_set = data_provider(self.args)
            val_set, test_set = random_split(test_set, [0, len(test_set)])
        else :
            dataset = data_provider(self.args)
            total_samples = len(dataset)
            train_size = int(0.8 * total_samples)
            val_size = 0
            test_size = total_samples - train_size
            print('test_size:',test_size)
            print('train_size:',train_size)
            # 随机划分数据集
            train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        # 创建 DataLoader
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=False,drop_last=True)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False,drop_last=True)
        return train_set,train_loader,val_set,val_loader,test_set,test_loader
    

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        time_now = time.time()
        train_steps = len(self.train_loader)
        # patience 默认是3，即连续3次的val_loss不下降就停止训练
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器 Adam
        model_optim = self._select_optimizer()
        # 选择损失函数 MSELoss
        criterion = self._select_criterion()

        # 如果使用混合精度训练
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # train_epoch 默认是10
        all_outputs=[]
        all_nwp=[]
        all_label=[]
        date_time = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            len_list = []

            for i, (batch_x, batch_y,nwp,date) in enumerate(self.train_loader):
                # print("{},{},{}".format(i,len(batch_x),batch_x.shape))
                iter_count += 1
                # 梯度清零
                model_optim.zero_grad()
                # 将数据转换为float类型
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.unsqueeze(2).expand(-1, -1, -1)
                batch_x_mark = None
                # nwp = nwp.unsqueeze(2).expand(-1,-1,-1)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # # 如果是PEMS或者Solar数据集，batch_x_mark和batch_y_mark都为None
                # if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                #     batch_x_mark = None
                #     batch_y_mark = None
                # else:
                #     # 否则将batch_x_mark和batch_y_mark转换为float类型
                #     batch_x_mark = batch_x_mark.float().to(self.device)
                #     batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # 将batch_y的后pred_len个时间步的数据全部置为0
                # pred_len  prediction sequence length
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # 将batch_y的前label_len个时间步的数据拼接到dec_inp后面
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, None)
                        # len_list.append(length)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    #  从batch_y中取出后pred_len个时间步的数据
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    #   计算loss
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    # Save outputs and date_time in the last epoch
                    if epoch == self.args.train_epochs - 1:
                        all_outputs.extend(outputs.cpu().detach().numpy())  # Save outputs
                        date_time.extend(date)  # Save date_time
                        all_label.extend(batch_y.cpu().detach().numpy())
                        all_nwp.extend(nwp)

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # print(self.model.mask_dwt.threshold_tensors[1].grad)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 暂时先不用验证
            train_loss = np.average(train_loss)
            print(f"Train Loss:{train_loss}")
            #     epoch + 1, train_steps, train_loss

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Save to CSV
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_label = np.concatenate(all_label,axis=0)
        all_nwp = np.concatenate(all_nwp,axis=0)
        self.save_to_csv(all_outputs, all_label,all_nwp, date_time,setting,True)
        return self.model

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        all_nwp = []
        date_time=[]
        folder_path = '/root/cloud/result/CSFformer/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y, nwp,batch_x_mark)in enumerate(self.test_loader):
            for i, (batch_x, batch_y, nwp, date)in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.unsqueeze(2).expand(-1, -1, -1)
                nwp = nwp.unsqueeze(2).expand(-1,-1,-1)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_mark = None
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                outputs = torch.clamp(outputs, min=0)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.extend(outputs)
                trues.extend(batch_y)
                all_nwp.extend(nwp)  # 记录时间信息
                date_time.extend(date)
                

        # 整理预测值和真实值
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        all_nwp = np.concatenate(all_nwp,axis=0)
        # 保存预测结果
        self.save_to_csv(preds, trues, all_nwp, date_time,setting, False)

        # 计算误差
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'mse: {mse}, mae: {mae}')
        print(f'settings:{setting}')
        # 保存误差结果
        result_file = os.path.join(folder_path, "result_long_term_forecast.txt")
        with open(result_file, 'a') as f:
            f.write(setting + "\n")
            f.write(f'mse: {mse}, mae: {mae}\n\n')

        return mae, mse, rmse, mape, mspe



    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def save_to_csv(self, preds, trues, all_nwp,date_time,setting, train):
        # Flatten the outputs and convert them into a DataFrame
       # 展平 preds, trues, 和 all_nwp
        # print(f"all_outputs length: {len(preds)}")
        # print(f"all_label length: {len(trues)}")
        # print(f"all_nwp length: {len(all_nwp)}")
        preds = preds.flatten() if isinstance(preds, np.ndarray) and preds.ndim > 1 else preds
        trues = trues.flatten() if isinstance(trues, np.ndarray) and trues.ndim > 1 else trues
        all_nwp = all_nwp.flatten() if isinstance(all_nwp, np.ndarray) and all_nwp.ndim > 1 else all_nwp

        data = {
            "label": trues,
            "output": preds,
            "fix_ghi":all_nwp,
            "date_time":date_time
        }
        
        df = pd.DataFrame(data)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Save to CSV file
        output_path = f"/root/cloud/data/afteritrans/{train}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
         # 确保父目录存在
        parent_dir = os.path.dirname(output_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        df.to_csv(output_path, index=False)
        print(f"Outputs and date_time saved to {output_path}")