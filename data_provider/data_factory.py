from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred

    elif flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False           # ✅ 关键：val/test 不丢 batch
        freq = args.freq
        batch_size = args.batch_size

    else:  # train
        shuffle_flag = True
        drop_last = True
        freq = args.freq
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )

    print(flag, len(data_set))

    # ✅ 关键：防止 batch_size > dataset 导致某些情况下 loader 为空（尤其你不小心又把 drop_last 打开时）
    if flag in ['test', 'val'] and len(data_set) > 0:
        batch_size = min(batch_size, len(data_set))
    elif flag in ['test', 'val'] and len(data_set) == 0:
        batch_size = 1  # 随便设一个，下面会在 test() 里报更清晰的错

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
