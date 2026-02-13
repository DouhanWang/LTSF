from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, LSTM, Naive, ARIMA
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, mean_WIS_interval, PICP

from utils.losses import quantile_loss  # import your custom loss
#from tabpfn_ts.models.tabpfn_ts import TabPFN_ts
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
print(">>> USING exp_main from:", __file__)

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.global_sigma = None # sd
        self.res_q_low = None   # ✅ 保存每个 horizon 的 q10 residual
        self.res_q_high = None  # ✅ 保存每个 horizon 的 q90 residual

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'LSTM': LSTM,
            'Naive': Naive,
            'ARIMA': ARIMA,
            #'TabPFN_ts': TabPFN_ts,
        }
        model = model_dict[self.args.model].Model(self.args).float()


        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        #criterion = nn.MSELoss()
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {self.args.loss}")
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_preds = []
        all_trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
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
                all_preds.append(pred.numpy())  # Add this
                all_trues.append(true.numpy())  # Add this
        total_loss = np.average(total_loss)
        all_preds = np.concatenate(all_preds, axis=0)  # Add this
        all_trues = np.concatenate(all_trues, axis=0)  # Add this
        self.model.train()
        return total_loss, all_preds, all_trues

    def _inverse_3d(self, dataset, arr3d):
        """arr3d: (B, L, C) -> inverse_transform per feature"""
        B, L, C = arr3d.shape
        flat = arr3d.reshape(-1, C)
        inv = dataset.inverse_transform(flat)
        return inv.reshape(B, L, C)

    def _calibrate_residual_quantiles(self, vali_data, vali_loader, alpha=0.2):
        """
        用 validation 的所有窗口、所有 horizon 残差，计算分位数残差区间：
          q_low[h] = quantile(residual_h, alpha/2)
          q_high[h] = quantile(residual_h, 1-alpha/2)
        返回 q_low, q_high，长度 pred_len
        """
        pred_len = self.args.pred_len
        residuals = [[] for _ in range(pred_len)]

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input（跟你 vali/test 保持一致）
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward
                if 'Linear' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention:
                        outputs = outputs[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                true = batch_y[:, -pred_len:, f_dim:]

                pred_np = outputs.detach().cpu().numpy()
                true_np = true.detach().cpu().numpy()

                # ✅ 统一到真实尺度后再算 residual（非常关键）
                if vali_data.scale:
                    pred_np = self._inverse_3d(vali_data, pred_np)
                    true_np = self._inverse_3d(vali_data, true_np)

                # ✅ 只取 target（最后一列）
                pred_t = pred_np[..., -1]  # (B, pred_len)
                true_t = true_np[..., -1]  # (B, pred_len)

                for h in range(pred_len):
                    residuals[h].append(true_t[:, h] - pred_t[:, h])

        self.model.train()

        q_low = np.zeros(pred_len, dtype=float)
        q_high = np.zeros(pred_len, dtype=float)

        for h in range(pred_len):
            if len(residuals[h]) == 0:
                # fallback：如果某个 horizon 没数据，设为 0（很少发生）
                q_low[h] = 0.0
                q_high[h] = 0.0
                continue
            r = np.concatenate(residuals[h], axis=0)  # (N,)
            q_low[h] = float(np.quantile(r, alpha / 2.0))
            q_high[h] = float(np.quantile(r, 1.0 - alpha / 2.0))

        return q_low, q_high


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # ==========================
        # 记录每个epoch的loss，用于画history曲线
        # ==========================
        train_loss_history = []
        vali_loss_history = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print("Start iterating train_loader...")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
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
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # change , batch_y
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_history.append(train_loss)

            if not self.args.train_only:
                # vali_loss = self.vali(vali_data, vali_loader, criterion)
                # test_loss = self.vali(test_data, test_loader, criterion)
                vali_loss, _, _ = self.vali(vali_data, vali_loader, criterion)  # Update this line
                vali_loss_history.append(vali_loss)
                # test_loss, _, _ = self.vali(test_data, test_loader, criterion)
                # 旧逻辑：每个epoch都在test上算loss；现在不需要，避免训练过程中反复查看test

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss)) #Test Loss: {4:.7f} , test_loss
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        # ==========================
        # 保存history plot到 logs/LookBackWindow/historyplot/
        # ==========================
        try:
            history_dir = os.path.join('logs', 'LookBackWindow', 'historyplot')
            os.makedirs(history_dir, exist_ok=True)

            plt.figure()
            plt.plot(train_loss_history, label='train_loss')
            if len(vali_loss_history) > 0:
                plt.plot(vali_loss_history, label='val_loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()

            # 按你指定的命名规则
            fname = f"{self.args.model}_simulated_Italy_ili_S_incidenza_sdscaler_uncertainty_{self.args.seq_len}_{self.args.moving_avg}.png"
            plt.savefig(os.path.join(history_dir, fname), dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved history plot: {os.path.join(history_dir, fname)}")
        except Exception as e:
            print(f"Warning: could not save history plot. Error: {e}")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # --- NEW BLOCK TO CALCULATE SIGMA ---
        print("Calculating global sigma from validation set residuals...")
        # ✅ NEW: 用 validation 残差分位数校准 80% 区间
        if not self.args.train_only:
            try:
                print("Calibrating residual quantiles on validation set for WIS(80%)...")
                q_low, q_high = self._calibrate_residual_quantiles(vali_data, vali_loader, alpha=0.2)
                self.res_q_low = q_low
                self.res_q_high = q_high
                print("Residual quantiles (q10/q90) by horizon:", list(zip(q_low, q_high)))


            except Exception as e:
                print(f"Warning: residual quantile calibration failed: {e}")
                self.res_q_low = None
                self.res_q_high = None
        else:
            self.res_q_low = None
            self.res_q_high = None

        # if not self.args.train_only:
        #     # Rerun validation on the best model
        #     vali_loss, vali_preds, vali_trues = self.vali(vali_data, vali_loader, criterion)
        #     # Calculate errors (residuals)
        #     errors = vali_trues - vali_preds
        #     # Calculate standard deviation of errors
        #     self.global_sigma = np.std(errors)  # This is the SCALED sigma
        #     scaled_sigma = self.global_sigma
        #
        #     # --- ADDED: Reverscale sigma for logging ---
        #     real_sigma = 0
        #     if vali_data.scale:
        #         try:
        #             if hasattr(vali_data.scaler, 'min_'):  # MinMaxScaler
        #                 target_scale = vali_data.scaler.scale_[-1]
        #                 real_sigma = self.global_sigma / target_scale
        #             elif hasattr(vali_data.scaler, 'mean_'):  # StandardScaler
        #                 target_scale = vali_data.scaler.scale_[-1]  # std
        #                 real_sigma = self.global_sigma * target_scale
        #         except Exception as e:
        #             print(f"Warning: could not reverscale global sigma for logging. Error: {e}")
        #             real_sigma = scaled_sigma  # fallback
        #     else:
        #         real_sigma = scaled_sigma  # It's already in real units
        #
        #     print(f"Global sigma (std dev of scaled residuals) calculated: {scaled_sigma:.6f}")
        #     print(f"Global sigma (REVERSCALED) calculated: {real_sigma:.6f}")
        #     # --- END OF ADDED BLOCK ---
        #
        # else:
        #     print("train_only=True. Cannot calculate validation sigma. Uncertainty plots will not be available.")
        #     self.global_sigma = 0  # Set to 0 to avoid errors
        # --- END OF NEW BLOCK ---
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, _ = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # ---------- folders ----------
        plot_folder = os.path.join('./test_results/', setting)
        os.makedirs(plot_folder, exist_ok=True)

        save_folder = os.path.join('./results/', setting)
        os.makedirs(save_folder, exist_ok=True)

        # ---------- 2) STITCH ONLY THE PREDICTIONS ----------
        # test_target_len = len(test_data.data_x) - seq_len  # e.g. 20
        # num_windows, _, C = preds.shape
        #
        # stitched_pred = np.full((test_target_len, C), np.nan, dtype=float)
        #
        # for w in range(num_windows):
        #     for h in range(pred_len):
        #         pos = w + h
        #         if pos < test_target_len and np.isnan(stitched_pred[pos]).all():
        #             stitched_pred[pos] = preds[w, h]
        # ==========================
        # ==========================
        # NEW: horizon-wise evaluation with extended forecast origins
        # allow windows whose 4-step prediction goes beyond season end,
        # but only keep targets within season for evaluation
        # ==========================
        # ---------- basic sizes ----------
        T = len(test_data.data_x)
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        C = test_data.data_x.shape[1]  # feature dims (S -> 1)

        # -------- get full truth in real scale (target incidenza) --------
        full_truth = test_data.data_x
        if test_data.scale:
            full_truth = test_data.inverse_transform(full_truth)
        full_truth_1d = full_truth[:, -1]  # target column (incidenza)

        # use full dates for plotting
        dates_window = test_data.dates.values  # length = T
        alpha = 0.2  # 80% interval

        # helper: build one model forward for a given start index s
        def forward_one_window(s):
            # encoder input
            end = s + seq_len

            if self.args.model == 'ARIMA':
                # Use within-season history for ARIMA (proper ARIMA), not only seq_len points.
                known0 = 4
                obs_end = min(known0 + s, len(test_data.data_x))  # test 已知到哪（最多到 season 末尾）

                # 1) 拼历史值
                x_enc = np.vstack([train_data.data_x, test_data.data_x[:obs_end]])  # (L_hist, C)

                # 2) 拼对应时间特征（和 x_enc 完全同长度）
                x_mark = np.vstack([train_data.data_stamp, test_data.data_stamp[:obs_end]])  # (L_hist, mark_dim)

            else:
                # other models keep strict short window
                x_enc = test_data.data_x[s:end]  # scaled, length = seq_len
                x_mark = test_data.data_stamp[s:end]
            # decoder marks need label_len+pred_len
            r_begin = s + seq_len - self.args.label_len
            r_end = r_begin + self.args.label_len + pred_len

            # 如果 r_end 超过 T（越界），我们就用最后一个 stamp 继续“复制”时间特征（简化）
            # 更严谨是按 weekly dates 外推再算 time_features，但复制在你的 timeF embedding 下通常也能跑通。
            if r_end <= len(test_data.data_stamp):
                y_mark = test_data.data_stamp[r_begin:r_end]
            else:
                y_mark = np.zeros((self.args.label_len + pred_len, test_data.data_stamp.shape[1]), dtype=float)
                # 前面 label_len 用真实 stamp
                y_mark[:self.args.label_len] = test_data.data_stamp[r_begin: r_begin + self.args.label_len]
                # 未来 pred_len 简单复制最后一行 stamp（够你先跑通逻辑）
                y_mark[self.args.label_len:] = test_data.data_stamp[-1]

            # decoder input: label part 用真实 y（scaled），未来 pred_len 用 0
            y_true_part = test_data.data_x[r_begin:r_begin + self.args.label_len]
            dec_zeros = np.zeros((pred_len, C), dtype=float)
            dec_inp = np.concatenate([y_true_part, dec_zeros], axis=0)

            # to torch
            batch_x = torch.from_numpy(x_enc).float().unsqueeze(0).to(self.device)  # (1, seq_len, C)
            batch_x_mark = torch.from_numpy(x_mark).float().unsqueeze(0).to(self.device)  # (1, seq_len, mark_dim)
            dec_inp_t = torch.from_numpy(dec_inp).float().unsqueeze(0).to(self.device)  # (1, label+pred, C)
            batch_y_mark = torch.from_numpy(y_mark).float().unsqueeze(0).to(self.device)  # (1, label+pred, mark_dim)

            with torch.no_grad():
                if 'Linear' in self.args.model:
                    out = self.model(batch_x)
                else:
                    out = self.model(batch_x, batch_x_mark, dec_inp_t, batch_y_mark)
                    if self.args.output_attention:
                        out = out[0]

            # slice last pred_len
            f_dim = -1 if self.args.features == 'MS' else 0
            out = out[:, -pred_len:, f_dim:]  # (1, pred_len, C')
            out_np = out.detach().cpu().numpy()[0]  # (pred_len, C')

            # ---- ARIMA CI hook: grab model-produced CI (same window) ----
            use_arima_ci = (self.args.model == 'ARIMA') and hasattr(self.model, "last_lower") and (
                        self.model.last_lower is not None)
            if use_arima_ci:
                # last_lower/upper were set inside ARIMA forward()
                lower_np = self.model.last_lower  # expected (B, pred_len, C)
                upper_np = self.model.last_upper

                # keep consistent slice and batch index 0
                # (ARIMA forward already outputs pred_len, but we slice defensively)
                lower_np = lower_np[0, -pred_len:, :]
                upper_np = upper_np[0, -pred_len:, :]
            else:
                lower_np, upper_np = None, None

            return out_np, lower_np, upper_np

        # -------- per-horizon evaluation --------
        results_by_h = {}

        for h in range(pred_len):  # h=0..3 => 1..4-step
            step = h + 1
            preds_h = []
            trues_h = []
            idx_h = []  # global time indices of targets
            arima_low_h = []  # NEW
            arima_up_h = []  # NEW

            # start index s: allow up to T-seq_len (not restricted by pred_len)
            for s in range(0, T - seq_len + 1):
                out, arima_low, arima_up = forward_one_window(s)

                target_global_idx = s + seq_len + h  # the time index we evaluate
                if target_global_idx >= T:
                    continue  # beyond season end -> do not evaluate

                # pick only horizon h
                pred_scaled = out[h, -1]  # target feature
                true_real = full_truth_1d[target_global_idx]

                preds_h.append(pred_scaled)
                trues_h.append(true_real)
                idx_h.append(target_global_idx)
                # NEW: store ARIMA CI (scaled) if available
                if self.args.model == 'ARIMA' and arima_low is not None:
                    arima_low_h.append(arima_low[h, -1])
                    arima_up_h.append(arima_up[h, -1])

            preds_h = np.array(preds_h, dtype=float)
            trues_h = np.array(trues_h, dtype=float)
            arima_low_h = np.array(arima_low_h, dtype=float) if len(arima_low_h) > 0 else None
            arima_up_h = np.array(arima_up_h, dtype=float) if len(arima_up_h) > 0 else None

            # inverse scale preds if needed (only for target)
            if test_data.scale:
                try:
                    if hasattr(test_data.scaler, 'min_'):
                        target_scale = test_data.scaler.scale_[-1]
                        target_min = test_data.scaler.min_[-1]
                        preds_h = (preds_h / target_scale) + target_min
                    elif hasattr(test_data.scaler, 'mean_'):
                        target_scale = test_data.scaler.scale_[-1]
                        target_mean = test_data.scaler.mean_[-1]
                        preds_h = (preds_h * target_scale) + target_mean
                except Exception as e:
                    print(f"Warning: could not reverscale preds_h. Error: {e}")

                # ---- choose CI source ----
                if self.args.model == 'ARIMA' and (arima_low_h is not None) and (arima_up_h is not None):
                    # ARIMA CI also needs inverse scaling (same as preds_h)
                    if test_data.scale:
                        try:
                            if hasattr(test_data.scaler, 'min_'):
                                target_scale = test_data.scaler.scale_[-1]
                                target_min = test_data.scaler.min_[-1]
                                arima_low_h = (arima_low_h / target_scale) + target_min
                                arima_up_h = (arima_up_h / target_scale) + target_min
                            elif hasattr(test_data.scaler, 'mean_'):
                                target_scale = test_data.scaler.scale_[-1]
                                target_mean = test_data.scaler.mean_[-1]
                                arima_low_h = (arima_low_h * target_scale) + target_mean
                                arima_up_h = (arima_up_h * target_scale) + target_mean
                        except Exception as e:
                            print(f"Warning: could not reverscale ARIMA CI. Error: {e}")

                    lower_h = arima_low_h
                    upper_h = arima_up_h
                else:
                    # your existing residual-quantile CI
                    lower_h = preds_h + float(self.res_q_low[h])
                    upper_h = preds_h + float(self.res_q_high[h])

            # else:
            #     real_sigma = self.global_sigma if self.global_sigma is not None else 0.0
            #     if test_data.scale:
            #         try:
            #             if hasattr(test_data.scaler, 'min_'):
            #                 real_sigma = real_sigma / test_data.scaler.scale_[-1]
            #             elif hasattr(test_data.scaler, 'mean_'):
            #                 real_sigma = real_sigma * test_data.scaler.scale_[-1]
            #         except Exception as e:
            #             print(f"Warning: could not reverscale sigma. Error: {e}")
            #     lower_h = preds_h - 1.2816 * real_sigma
            #     upper_h = preds_h + 1.2816 * real_sigma

            lower_h = np.clip(lower_h, a_min=0, a_max=None)
            # ---- NEW: per-point WIS (80% interval), for boxplots ----
            # WIS_alpha(l,u,y) = (u-l) + (2/alpha)*(l-y)*1[y<l] + (2/alpha)*(y-u)*1[y>u]
            width = upper_h - lower_h
            under = np.maximum(lower_h - trues_h, 0.0)
            over = np.maximum(trues_h - upper_h, 0.0)
            wis80_point = width + (2.0 / alpha) * (under + over)  # shape [N]
            np.save(os.path.join(save_folder, f"wis80_point_step{step}.npy"), wis80_point)

            picp80 = PICP(lower_h, upper_h, trues_h)
            wis80 = mean_WIS_interval(lower_h, upper_h, trues_h, alpha)

            from utils.metrics import metric
            point_metrics = metric(preds_h.reshape(1, -1, 1), trues_h.reshape(1, -1, 1))

            print(f"\n===== {h + 1}-step (save only step {h + 1}) =====")
            print("PICP (80% interval):", picp80)
            print("Mean WIS (80% interval):", wis80)
            print("Point metrics:", point_metrics)

            # -------- plot: put predictions back to full timeline --------
            pred_plot = np.full(T, np.nan, dtype=float)
            lower_plot = np.full(T, np.nan, dtype=float)
            upper_plot = np.full(T, np.nan, dtype=float)

            for j, gidx in enumerate(idx_h):
                pred_plot[gidx] = preds_h[j]
                lower_plot[gidx] = lower_h[j]
                upper_plot[gidx] = upper_h[j]

            import pandas as pds
            gt_s = pds.Series(full_truth_1d, index=dates_window)
            pred_s = pds.Series(pred_plot, index=dates_window)
            lower_s = pds.Series(lower_plot, index=dates_window)
            upper_s = pds.Series(upper_plot, index=dates_window)

            visual(true=gt_s,
                   preds=pred_s,
                   path=os.path.join(plot_folder, f"rolling_test_step{h + 1}.png"),
                   lower=lower_s,
                   upper=upper_s,
                   seq_len=seq_len)

            # ---- SAVE per-step npy ----
            np.save(os.path.join(save_folder, f"pred_step{step}.npy"), preds_h)
            np.save(os.path.join(save_folder, f"true_step{step}.npy"), trues_h)
            np.save(os.path.join(save_folder, f"lower80_step{step}.npy"), lower_h)
            np.save(os.path.join(save_folder, f"upper80_step{step}.npy"), upper_h)
            np.save(os.path.join(save_folder, f"idx_step{step}.npy"), np.array(idx_h, dtype=int))

            # ---- SAVE per-step CSV (date-aligned, NaN elsewhere) ----
            df_step = pd.DataFrame({
                "date": pd.to_datetime(dates_window),
                "true": full_truth_1d,
                f"pred_step{step}": pred_plot,
                f"lower80_step{step}": lower_plot,
                f"upper80_step{step}": upper_plot,
            })
            df_step.to_csv(os.path.join(save_folder, f"rolling_pred_step{step}.csv"), index=False)

            # ---- SAVE metrics line (append) ----
            metrics_line = {
                "step": step,
                "PICP80": float(picp80),
                "WIS80": float(wis80),
                **{k: float(v) for k, v in point_metrics.items()}
            }
            with open(os.path.join(save_folder, "metrics_by_step.txt"), "a", encoding="utf-8") as f:
                f.write(setting + "  " + "  ".join([f"{k}:{v:.6f}" for k, v in metrics_line.items()]) + "\n")

            results_by_h[step] = metrics_line

        return




    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
    #
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #
    #     preds = []
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input
    #             dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if 'Linear' in self.args.model:
    #                         outputs = self.model(batch_x)
    #                     else:
    #                         if self.args.output_attention:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                         else:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if 'Linear' in self.args.model:
    #                     outputs = self.model(batch_x)
    #                 else:
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             pred = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(pred)
    #
    #     preds = np.array(preds)
    #     preds = np.concatenate(preds, axis=0)
    #     if (pred_data.scale):
    #         preds = pred_data.inverse_transform(preds)
    #
    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     np.save(folder_path + 'real_prediction.npy', preds)
    #     pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)
    #
    #     return
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
                # --- UPDATED: Handle Quantile vs. MSE decoder input ---
                if self.args.loss == 'quantile':
                    num_quantiles = len(self.args.quantiles)
                    zeros_shape = (batch_y.shape[0], self.args.pred_len, self.args.dec_in)
                    dec_inp_zeros = torch.zeros(zeros_shape).float().to(self.device)
                    label_part = batch_y[:, :self.args.label_len, :]
                    dec_inp_label = label_part.repeat(1, 1, num_quantiles)
                    dec_inp = torch.cat([dec_inp_label, dec_inp_zeros], dim=1).float().to(self.device)
                else:
                    # Original logic for MSE/MAE
                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                        batch_y.device)
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # --- END OF UPDATE ---

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[
                                0] if self.args.output_attention else self.model(batch_x, batch_x_mark, dec_inp,
                                                                                 batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[
                            0] if self.args.output_attention else self.model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark)

                # --- UPDATED: Handle Quantile vs. MSE output slicing ---
                # We need to slice here, *before* saving, to get the right shape
                f_dim = -1 if self.args.features == 'MS' else 0
                if self.args.loss == 'quantile' and self.args.features == 'MS':
                    num_quantiles = len(self.args.quantiles)
                    outputs = outputs[:, -self.args.pred_len:, -num_quantiles:]
                elif self.args.loss != 'quantile':
                    # This is the critical part for MS/MSE
                    # We get ALL features if M, but only last feature if MS
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                else:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # --- END OF UPDATE ---

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)

        # --- NEW CODE BLOCK FOR UNCERTAINTY ---

        # We rename 'preds' to 'preds_mean' for clarity
        preds_mean = preds
        preds_lower = None
        preds_upper = None

        # Check if we have a sigma to work with
        if self.global_sigma is None:
            print("Warning: self.global_sigma not calculated. Run training first (is_training=1).")
            self.global_sigma = 0  # Set to 0 to avoid errors

        # Reverscale the mean predictions
        if (pred_data.scale):
            print("Reverscaling mean predictions...")
            preds_mean = pred_data.inverse_transform(preds_mean)

        # Calculate uncertainty bands (only if loss is not quantile)
        if (self.args.loss != 'quantile' and pred_data.scale):
            print("Calculating reverscaled uncertainty bands...")
            try:
                real_sigma = 0
                # Get the scaling parameters for the *last feature* (incidenza)
                if hasattr(pred_data.scaler, 'min_'):  # MinMaxScaler
                    target_scale = pred_data.scaler.scale_[-1]
                    real_sigma = self.global_sigma / target_scale
                elif hasattr(pred_data.scaler, 'mean_'):  # StandardScaler
                    target_scale = pred_data.scaler.scale_[-1]  # std
                    real_sigma = self.global_sigma * target_scale

                # Get the mean prediction for the last feature (incidenza)
                # preds_mean shape is (N_samples, pred_len, 1) if MS/MSE
                mean_incidenza = preds_mean[..., -1]  # Ellipsis (...) means "all prior dimensions"

                # Calculate bands (shape will be N_samples, pred_len)
                preds_lower = mean_incidenza - 1.96 * real_sigma
                preds_upper = mean_incidenza + 1.96 * real_sigma

                # Clip lower bound at 0 (incidenza can't be negative)
                preds_lower = np.clip(preds_lower, a_min=0, a_max=None)

            except Exception as e:
                print(f"Error calculating uncertainty bands: {e}")

        # --- END OF NEW CODE BLOCK ---

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # --- UPDATED: Save mean, lower, and upper .npy files ---
        print("Saving .npy files...")
        np.save(folder_path + 'real_prediction_mean.npy', preds_mean)
        if preds_lower is not None:
            np.save(folder_path + 'real_prediction_lower.npy', preds_lower)
        if preds_upper is not None:
            np.save(folder_path + 'real_prediction_upper.npy', preds_upper)

        # --- UPDATED: Save comprehensive .csv file ---
        print("Saving .csv file...")
        try:
            # We save only the first sample from the prediction batch (preds_mean[0])
            # This is standard for this codebase

            # Get the column names for features (e.g., ['assistiti', 'incidenza'])
            feature_cols = [col for col in pred_data.cols if col != 'date']

            # Create a DataFrame with the mean prediction(s)
            # preds_mean[0] has shape (pred_len, num_features)
            df = pd.DataFrame(preds_mean[0], columns=feature_cols)

            # Add the future dates
            df.insert(0, 'date', pred_data.future_dates)

            # If we have uncertainty, add it
            if preds_lower is not None and preds_upper is not None:
                # We need the bands from the first sample
                df['incidenza_lower_95'] = preds_lower[0]
                df['incidenza_upper_95'] = preds_upper[0]
                # We can also rename the incidenza column for clarity
                df.rename(columns={'incidenza': 'incidenza_mean'}, inplace=True)

            df.to_csv(folder_path + 'real_prediction.csv', index=False)

        except Exception as e:
            print(f"Error saving .csv file: {e}")
            # Fallback to original save logic just in case
            if self.args.loss == 'quantile':
                print("CSV saving for quantiles is not fully implemented in this update.")
            else:
                pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds_mean[0], axis=1),
                             columns=['date'] + feature_cols).to_csv(folder_path + 'real_prediction_fallback.csv',
                                                                     index=False)
        # --- END OF UPDATES ---

        return