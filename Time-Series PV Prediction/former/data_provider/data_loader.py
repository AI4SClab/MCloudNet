import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):

        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        # 计算IQR并检测异常值
        # Q1 = df_data.quantile(0.25)
        # Q3 = df_data.quantile(0.75)
        # IQR = Q3 - Q1

        # 定义异常值的条件
        # is_outlier = (df_data < (Q1 - 1.5 * IQR)) | (df_data > (Q3 + 1.5 * IQR))
        
        # 计算均值和标准差并检测异常值
        # mean = df_data.mean()
        # std = df_data.std()

        # 定义异常值的条件，这里我们使用超出均值加减3个标准差的范围作为异常值的标准
        # is_outlier = (df_data < (mean - 3 * std)) | (df_data > (mean + 3 * std))

        # 对于异常值，使用线性插值进行填补
        # df_interpolated = df_data.copy()
        # for column in df_data.columns:
            # 只对存在异常值的列进行插值
        #     if is_outlier[column].any():
                # 使用线性插值填补
        #         df_data[column].interpolate(method='linear', inplace=True, limit_direction='both')

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Cloud_numonly3(torch.utils.data.Dataset):
    def __init__(self, root_path, data_path,size, freq='15min'):

        self.data =  pd.read_csv(os.path.join(root_path,data_path))
        # 保留原始索引
        self.filtered_data = self.data.dropna(subset=['Output_ghi']).reset_index()
        #self.filtered_data = self.data.reset_index()
        # self.lastrow = 8000

        self.lmd_columns = [
            'nwp_globalirrad',
            'nwp_directirrad',
            'nwp_temperature',
            'nwp_humidity',
            'nwp_windspeed',
            'nwp_winddirection',
            'nwp_pressure',
            'lmd_totalirrad',
            'lmd_diffuseirrad',
            'lmd_temperature',
            'lmd_pressure',
            'lmd_winddirection',
            'lmd_windspeed',
            'power'
        ]
        self.freq = freq
        self.input_len = size[0]
        self.pred_len = size[1]
        self.samples = self._create_samples()  # Preprocess to create samples

    def inverse_transform(self, normalized_data):
        # 使用 scaler 的属性进行反归一化
        scaler = StandardScaler()
        return normalized_data * self.scaler.scale_ + self.scaler.mean_

    def _create_samples(self):
        samples = []
        total_rows = len(self.filtered_data)
        # Iterate with a sliding window over the data
        for _, row in self.filtered_data.iterrows():
            input_data_x = []
            date_mark = []
            end = row['index']  # 原始数据中的索引(作为标签)
            i = end - self.input_len
            # 带有修正nwpghi信息的行
            while i < end:
                seqrow = self.data.iloc[i]
                # 处理lmd信息
                lmd_features = [seqrow[col] for col in self.lmd_columns]
                lmd_tensor = torch.tensor(lmd_features, dtype=torch.float32)
                # 将每一个 combined_features 作为 input_data_x 的一行
                input_data_x.append(lmd_tensor)
                
                # timestamp = pd.to_datetime(seqrow['date_time'])  # 当前行的时间戳
                # df_stamp = pd.DataFrame(columns=['date_time'])
                # df_stamp['date_time'] = [timestamp]
                # date_stamp = time_features(pd.to_datetime(df_stamp['date_time'].values), freq=self.freq)
                # # date_stamp = date_stamp.transpose(1, 0)
                # date_stamp = torch.tensor(date_stamp).squeeze(1)
                # date_mark.append(date_stamp)

                i = i + 1

            # date_tensor = torch.stack(date_mark)
            input_data_x_tensor = torch.stack(input_data_x)
            # Label
            output_data_y = torch.tensor(row['power'], dtype=torch.float32)
            output_data_y = [output_data_y]
            date_tensor = row['date_time'].strftime("%Y-%m-%d_%H-%M-%S") if isinstance(row['date_time'], pd.Timestamp) else str(row['date_time'])
            nwp = torch.tensor(row['Output_ghi'], dtype=torch.float32)
            nwp = [nwp]
            # samples.append((input_data_x_tensor, torch.stack(output_data_y), torch.stack(nwp),date_tensor))
            samples.append((input_data_x_tensor, torch.stack(output_data_y),torch.stack(nwp),date_tensor))
        print(f"i:{i}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_data_x, output_data_y,nwp,date_tensor = self.samples[idx]
        return input_data_x, output_data_y,nwp,date_tensor

    def save_to_csv(self, output_path):
        """
        将数据集保存到 CSV 文件。
        """
        rows = []
        
        # 遍历样本
        for input_data_x, output_data_y,in self.samples:
            # 将 input_data_x 转换为列表
            input_data_x_list = input_data_x.numpy().flatten().tolist()
            # 将 output_data_y 转换为列表
            output_data_y_list = output_data_y.numpy().tolist()
            
            # 将 input_data_x 和 output_data_y 合并为一行
            row = input_data_x_list + output_data_y_list
            rows.append(row)
        
        # 定义 CSV 文件的列名
        column_names = [f"{col}_{i}" for col in self.lmd_columns for i in range(self.input_len)] + ['power']
        
        # 将数据写入 CSV 文件
        df = pd.DataFrame(rows, columns=column_names)
        df.to_csv(output_path, index=False)


class Dataset_Cloud_numonly4(torch.utils.data.Dataset):
    def __init__(self, root_path, data_path,size, freq='15min'):

        self.data =  pd.read_csv(os.path.join(root_path,data_path))
        # 保留原始索引
        self.filtered_data = self.data.dropna(subset=['Output_ghi']).reset_index()

        self.lmd_columns = [
            'nwp_globalirrad',
            'nwp_directirrad',
            'nwp_temperature',
            'nwp_humidity',
            'nwp_windspeed',
            'nwp_winddirection',
            'nwp_pressure',
            'lmd_totalirrad',
            'lmd_diffuseirrad',
            'lmd_temperature',
            'lmd_pressure',
            'lmd_winddirection',
            'lmd_windspeed',
            'power'
        ]
        self.freq = freq
        self.input_len = size[0]
        self.pred_len = size[1]
        self.samples = self._create_samples()  # Preprocess to create samples

    def inverse_transform(self, normalized_data):
        # 使用 scaler 的属性进行反归一化
        scaler = StandardScaler()
        return normalized_data * self.scaler.scale_ + self.scaler.mean_

    def _create_samples(self):
        samples = []
        total_rows = len(self.filtered_data)
        # Iterate with a sliding window over the data
        for _, row in self.filtered_data.iterrows():
            input_data_x = []
            date_mark = []
            end = row['index']  # 原始数据中的索引(作为标签)
            i = end - self.input_len
            # 带有修正nwpghi信息的行
            while i < end:
                seqrow = self.data.iloc[i]
                # 处理lmd信息
                lmd_features = [seqrow[col] for col in self.lmd_columns]
                lmd_tensor = torch.tensor(lmd_features, dtype=torch.float32)
                # 将每一个 combined_features 作为 input_data_x 的一行
                input_data_x.append(lmd_tensor)
                
                # timestamp = pd.to_datetime(seqrow['date_time'])  # 当前行的时间戳
                # df_stamp = pd.DataFrame(columns=['date_time'])
                # df_stamp['date_time'] = [timestamp]
                # date_stamp = time_features(pd.to_datetime(df_stamp['date_time'].values), freq=self.freq)
                # # date_stamp = date_stamp.transpose(1, 0)
                # date_stamp = torch.tensor(date_stamp).squeeze(1)
                # date_mark.append(date_stamp)

                i = i + 1

            #date_tensor = torch.stack(date_mark)
            input_data_x_tensor = torch.stack(input_data_x)
            # Label
            output_data_y = torch.tensor(row['power'], dtype=torch.float32)
            output_data_y = [output_data_y]
            label_date = torch.tensor(row['date_time'], dtype=torch.float32)
            label_date = [label_date]
            nwp = torch.tensor(row['Output_ghi'], dtype=torch.float32)
            nwp = [nwp]
            samples.append((input_data_x_tensor, torch.stack(output_data_y), torch.stack(nwp),label_date))
        print(f"i:{i}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        end = idx['index']
        output_data_x = torch.tensor(self.data.loc[idx:idx+self.pred_len-1, 'power'].values, dtype=torch.float32)
        output_data_y = torch.tensor(self.filtered_data.loc[idx:idx+self.pred_len-1, 'power'].values, dtype=torch.float32)
        return input_data_x, output_data_y,nwp,date_tensor

    def save_to_csv(self, output_path):
        """
        将数据集保存到 CSV 文件。
        """
        rows = []
        
        # 遍历样本
        for input_data_x, output_data_y in self.samples:
            # 将 input_data_x 转换为列表
            input_data_x_list = input_data_x.numpy().flatten().tolist()
            # 将 output_data_y 转换为列表
            output_data_y_list = output_data_y.numpy().tolist()
            
            # 将 input_data_x 和 output_data_y 合并为一行
            row = input_data_x_list + output_data_y_list
            rows.append(row)
        
        # 定义 CSV 文件的列名
        column_names = [f"{col}_{i}" for col in self.lmd_columns for i in range(self.input_len)] + ['power']
        
        # 将数据写入 CSV 文件
        df = pd.DataFrame(rows, columns=column_names)
        df.to_csv(output_path, index=False)