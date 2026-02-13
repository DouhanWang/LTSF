import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
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
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False):
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


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', train_only=False):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.train_only = train_only
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         if self.features == 'S':
#             cols.remove(self.target)
#         cols.remove('date')
#         # print(cols)
#         num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             df_raw = df_raw[['date'] + cols]
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_raw = df_raw[['date'] + cols + [self.target]]
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             # print(self.scaler.mean_)
#             # exit()
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False,
                 real_only_val_test=False, real_only_scaler=False):
        self.real_only_val_test = real_only_val_test
        self.real_only_scaler = real_only_scaler
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
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()  # or MinMaxScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if "combined" in str(self.data_path).lower():
            self.real_only_val_test = True
            self.real_only_scaler = True
        else:
            # 保证旧实验完全不变（尤其是你没传这俩参数时）
            self.real_only_val_test = getattr(self, "real_only_val_test", False)
            self.real_only_scaler = getattr(self, "real_only_scaler", False)
        # ---------- ensure series_id exists ----------
        if 'item_id' not in df_raw.columns:
            # original single-series file
            df_raw['item_id'] = 0

        # ---------- for TEST split: keep only original series ----------
        # set_type: 0=train, 1=val, 2=test in your codebase
        # if self.set_type == 2:
        #     # only keep original series (series_id=0)
        #     df_raw = df_raw[df_raw['item_id'] == 0].reset_index(drop=True)
        # 旧逻辑：test只保留item_id=0；现在要测试1000条序列，所以不再过滤item_id




        df_raw['date_str'] = (
                df_raw['anno'].astype(str) + ' ' +
                df_raw['settimana'].astype(str) + ' 1'
        )
        df_raw['date'] = pd.to_datetime(df_raw['date_str'], format='%Y %W %w')

        # keep only the columns we need
        df_raw = df_raw[['item_id', 'season_id', 'date', 'incidenza']].copy() ## 原本没有season_id

        # # set target name for this dataset
        # self.target = 'incidenza'

        # sorted by series and date: critical
        df_raw = df_raw.sort_values(['item_id', 'date']).reset_index(drop=True)

        # -------- target / feature columns list --------
        cols = list(df_raw.columns)
        # df_raw: ['series_id', 'date', 'numero_assistiti', 'incidenza']
        cols.remove('item_id')
        cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)

        # -------- build df_data (numeric part) --------
        if self.features in ['M', 'MS']:
            df_data = df_raw[cols]
        else:  # 'S'
            df_data = df_raw[[self.target]]

        N = len(df_raw)
        # =========================
        # 新逻辑：按season划分数据
        # train: season_id in {0,1,2}
        # val  : season_id == 2
        # test : season_id == 3
        # 并且用 segment_id = item_id*10 + season_id 来确保窗口不跨season
        # =========================

        # --- ensure season_id exists ---
        if 'season_id' not in df_raw.columns:
            raise ValueError("你的数据里缺少 season_id 列（0/1/2/3）。建议在生成CSV时加入该列。")

        n_item = df_raw['item_id'].nunique()

        if n_item == 1:
            # ===== REAL DATA MODE: one long TS =====
            # Sort by date only (single series)
            df_raw = df_raw.sort_values(['date']).reset_index(drop=True)

            # Build df_data AFTER final sorting
            cols = list(df_raw.columns)
            cols.remove('item_id')
            cols.remove('season_id')
            cols.remove('date')
            if self.features == 'S':
                if self.target in cols:
                    cols.remove(self.target)

            if self.features in ['M', 'MS']:
                df_data = df_raw[cols]
            else:
                df_data = df_raw[[self.target]]

            N = len(df_raw)
            series_ids = df_raw['item_id'].values  # all zeros, single block

            # train = season 0/1/2 (concatenated), val = train, test = season 3
            train_mask = df_raw['season_id'].isin([0, 1, 2]).values
            val_mask = train_mask
            test_mask = (df_raw['season_id'] == 3).values

        else:
            # ===== SIMULATED MODE: each season independent =====
            # segment_id = item*10 + season, so windows do NOT cross seasons
            df_raw['segment_id'] = df_raw['item_id'].astype(int) * 10 + df_raw['season_id'].astype(int)
            df_raw = df_raw.sort_values(['segment_id', 'date']).reset_index(drop=True)

            # Build df_data AFTER final sorting
            cols = list(df_raw.columns)
            cols.remove('item_id')
            cols.remove('season_id')
            cols.remove('date')
            if 'segment_id' in cols:
                cols.remove('segment_id')
            if self.features == 'S':
                if self.target in cols:
                    cols.remove(self.target)

            if self.features in ['M', 'MS']:
                df_data = df_raw[cols]
            else:
                df_data = df_raw[[self.target]]

            N = len(df_raw)
            series_ids = df_raw['segment_id'].values

            # keep your original season split (you can keep your previous definition)
            train_mask = df_raw['season_id'].isin([0, 1]).values
            unique_items = np.sort(df_raw['item_id'].unique())
            median_item = int(unique_items[len(unique_items) // 2])  # e.g., 1000 items -> index 500

            if self.real_only_val_test:
                # keep old behavior: only item_id==0 for val/test
                val_item = 0
                test_item = 0
            else:
                # NEW behavior: use median item for val and test (simulated case)
                val_item = median_item
                test_item = median_item

            val_mask = ((df_raw['season_id'] == 2) & (df_raw['item_id'] == val_item)).values
            test_mask = ((df_raw['season_id'] == 3) & (df_raw['item_id'] == test_item)).values

        # ---------------------------
        # # # 旧逻辑：每条序列最后20点做test（现在不用）
        # # 旧逻辑：series_ids是item_id
        # series_ids = df_raw['item_id'].values
        # num_test_target = 20  # want last 20 points per series cut from train
        #
        # # ---------- 1) per-row train/test masks ----------
        # train_mask = np.zeros(N, dtype=bool)
        # test_mask = np.zeros(N, dtype=bool)
        #
        # for sid in np.unique(series_ids):
        #     idx = np.where(series_ids == sid)[0]
        #     n = len(idx)
        #
        #     if n <= num_test_target + self.seq_len:
        #         # too short: use all as train
        #         train_mask[idx] = True
        #         continue
        #
        #     if sid == 0:
        #         # ORIGINAL series: keep last 20 targets for test (with seq_len context)
        #         test_len = num_test_target + self.seq_len
        #         test_start_local = n - test_len
        #         test_idx = idx[test_start_local:]
        #         test_mask[test_idx] = True
        #
        #         # train for original: everything before last 20 targets
        #         train_end_local = n - num_test_target
        #         train_idx = idx[:train_end_local]
        #         train_mask[train_idx] = True
        #     else:
        #         # AUGMENTED series: cut last 20 points from train/test completely
        #         cutoff_local = max(0, n - num_test_target)
        #         train_idx = idx[:cutoff_local]
        #         train_mask[train_idx] = True
        #         # idx[cutoff_local:] (last 20) are ignored entirely

        # ---------- 2) choose split_idx based on flag ----------
        if self.train_only:
            split_idx = np.where(train_mask)[0]
        else:
            if self.set_type == 0:  # train
                split_idx = np.where(train_mask)[0]
            elif self.set_type == 1:  # val -> SAME AS TRAIN (no real val)
                split_idx = np.where(val_mask)[0]  ## 旧的是train_mask
            else:  # test -> only original series last seq_len+20
                split_idx = np.where(test_mask)[0]

        # ---------- 3) scale with train rows only ----------
        if self.scale:
            if self.real_only_scaler:
                scaler_mask = train_mask & (df_raw['item_id'].values == 0)
                train_idx_for_scaler = np.where(scaler_mask)[0]
            else:
                train_idx_for_scaler = np.where(train_mask)[0]
            train_data = df_data.iloc[train_idx_for_scaler]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ---------- 4) timestamps for this split ----------
        df_stamp = df_raw[['date']].iloc[split_idx].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        self.dates = df_stamp['date'].reset_index(drop=True)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # ---------- 5) apply split indices ----------
        self.data_x = data[split_idx]
        self.data_y = data[split_idx]
        self.data_stamp = data_stamp

        # store series_id for this split (to build windows that don't cross)
        self.series_split = series_ids[split_idx]

        # ---------- 6) precompute valid window starts ----------
        window_len = self.seq_len + self.pred_len
        valid_idx = []

        start = 0
        while start < len(self.series_split):
            sid = self.series_split[start]
            end = start
            while end < len(self.series_split) and self.series_split[end] == sid:
                end += 1
            block_len = end - start
            max_start_local = block_len - window_len + 1
            if max_start_local > 0:
                for local in range(max_start_local):
                    valid_idx.append(start + local)
            start = end

        self.valid_idx = np.array(valid_idx, dtype=int)

    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len
    #     r_end = r_begin + self.label_len + self.pred_len
    #
    #     seq_x = self.data_x[s_begin:s_end]
    #     seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]
    #
    #     return seq_x, seq_y, seq_x_mark, seq_y_mark
    def __getitem__(self, index):
        real_idx = self.valid_idx[index]

        s_begin = real_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    # def __len__(self):
    #     return len(self.data_x) - self.seq_len - self.pred_len + 1
    def __len__(self):
        return len(self.valid_idx)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False):
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
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
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
        self.future_dates = list(pred_dates[1:])
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
