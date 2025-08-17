from .data_loader import Dataset_Cloud_numonly3
from torch.utils.data import DataLoader

data_dict = {
#    'ETTh1': Dataset_ETT_hour,
#    'ETTh2': Dataset_ETT_hour,
#    'ETTm1': Dataset_ETT_minute,
#    'ETTm2': Dataset_ETT_minute,
#    'Solar': Dataset_Solar,
#    'PEMS': Dataset_PEMS,
#    'custom': Dataset_Custom,
    'cloud_numonly3':Dataset_Cloud_numonly3, # 有全部的数据 处理resnet的结果
}

def data_provider(args):
  Data = data_dict[args.data]
  timeenc = 0 if args.embed != 'timeF' else 1

  root_path =args.root_path
  data_path=args.data_path
  size=[args.seq_len, args.label_len, args.pred_len]
  imgsize=10
  test_path = args.data_test_path

  if args.isdiff:
    train_set = Data(
        root_path =root_path,
        data_path=data_path,
        size=size
    )
    test_set =Data(
        root_path =root_path,
        data_path=test_path,
        size=size
    )
    return train_set,test_set
  else:
    data_set = Data(
        root_path =root_path,
        data_path=data_path,
        size=size
        # features=features,
        # target=target,
        # timeenc=timeenc,
        # freq=freq,
    )
    return data_set