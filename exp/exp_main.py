from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
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

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.global_sigma = None # sd
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            #'TabPFN_ts': TabPFN_ts
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
        elif self.args.loss == 'quantile':
            # wrap quantile_loss into a lambda to pass quantiles from args
            def criterion(y_pred, y_true):
                return quantile_loss(y_true, y_pred, self.args.quantiles)
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

                # # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # decoder input
                if self.args.loss == 'quantile':
                    # For quantile loss, we must expand the decoder input to match dec_in (which is 21)
                    num_quantiles = len(self.args.quantiles)  # e.g., 3

                    # 1. Create the "zero" part with the correct quantile dimension
                    # Use dec_in here, which you set to 21 in the script
                    zeros_shape = (batch_y.shape[0], self.args.pred_len, self.args.dec_in)
                    dec_inp_zeros = torch.zeros(zeros_shape).float().to(self.device)

                    # 2. Create the "label" part by repeating the ground truth for each quantile
                    label_part = batch_y[:, :self.args.label_len, :]  # Shape: (batch, label_len, 7)

                    # Repeat tensor 'num_quantiles' times along the last dimension
                    # Shape becomes (batch, label_len, 7 * 3) = (batch, label_len, 21)
                    dec_inp_label = label_part.repeat(1, 1, num_quantiles)

                    # 3. Concatenate them
                    dec_inp = torch.cat([dec_inp_label, dec_inp_zeros], dim=1).float().to(self.device)

                else:
                    # Original logic
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
                if self.args.loss == 'quantile' and self.args.features == 'MS':
                    # We need the quantiles for the *target* feature only.
                    # Since target is the last feature, we take the last num_quantiles channels.
                    num_quantiles = len(self.args.quantiles)
                    outputs = outputs[:, -self.args.pred_len:, -num_quantiles:]  # Shape (batch, pred_len, 3)
                else:
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # decoder input
                if self.args.loss == 'quantile':
                    # For quantile loss, we must expand the decoder input to match dec_in (which is 21)
                    num_quantiles = len(self.args.quantiles)  # e.g., 3

                    # 1. Create the "zero" part with the correct quantile dimension
                    # Use dec_in here, which you set to 21 in the script
                    zeros_shape = (batch_y.shape[0], self.args.pred_len, self.args.dec_in)
                    dec_inp_zeros = torch.zeros(zeros_shape).float().to(self.device)

                    # 2. Create the "label" part by repeating the ground truth for each quantile
                    label_part = batch_y[:, :self.args.label_len, :]  # Shape: (batch, label_len, 7)

                    # Repeat tensor 'num_quantiles' times along the last dimension
                    # Shape becomes (batch, label_len, 7 * 3) = (batch, label_len, 21)
                    dec_inp_label = label_part.repeat(1, 1, num_quantiles)

                    # 3. Concatenate them
                    dec_inp = torch.cat([dec_inp_label, dec_inp_zeros], dim=1).float().to(self.device)

                else:
                    # Original logic
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
                    if self.args.loss == 'quantile' and self.args.features == 'MS':
                        # We need the quantiles for the *target* feature only.
                        # Since target is the last feature, we take the last num_quantiles channels.
                        num_quantiles = len(self.args.quantiles)
                        outputs = outputs[:, -self.args.pred_len:, -num_quantiles:]  # Shape (batch, pred_len, 3)
                    else:
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
            if not self.args.train_only:
                # vali_loss = self.vali(vali_data, vali_loader, criterion)
                # test_loss = self.vali(test_data, test_loader, criterion)
                vali_loss, _, _ = self.vali(vali_data, vali_loader, criterion)  # Update this line
                test_loss, _, _ = self.vali(test_data, test_loader, criterion)  # Update this line

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # --- NEW BLOCK TO CALCULATE SIGMA ---
        print("Calculating global sigma from validation set residuals...")
        if not self.args.train_only:
            # Rerun validation on the best model
            vali_loss, vali_preds, vali_trues = self.vali(vali_data, vali_loader, criterion)
            # Calculate errors (residuals)
            errors = vali_trues - vali_preds
            # Calculate standard deviation of errors
            self.global_sigma = np.std(errors)  # This is the SCALED sigma
            scaled_sigma = self.global_sigma

            # --- ADDED: Reverscale sigma for logging ---
            real_sigma = 0
            if vali_data.scale:
                try:
                    if hasattr(vali_data.scaler, 'min_'):  # MinMaxScaler
                        target_scale = vali_data.scaler.scale_[-1]
                        real_sigma = self.global_sigma / target_scale
                    elif hasattr(vali_data.scaler, 'mean_'):  # StandardScaler
                        target_scale = vali_data.scaler.scale_[-1]  # std
                        real_sigma = self.global_sigma * target_scale
                except Exception as e:
                    print(f"Warning: could not reverscale global sigma for logging. Error: {e}")
                    real_sigma = scaled_sigma  # fallback
            else:
                real_sigma = scaled_sigma  # It's already in real units

            print(f"Global sigma (std dev of scaled residuals) calculated: {scaled_sigma:.6f}")
            print(f"Global sigma (REVERSCALED) calculated: {real_sigma:.6f}")
            # --- END OF ADDED BLOCK ---

        else:
            print("train_only=True. Cannot calculate validation sigma. Uncertainty plots will not be available.")
            self.global_sigma = 0  # Set to 0 to avoid errors
        # --- END OF NEW BLOCK ---
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # decoder input
                if self.args.loss == 'quantile':
                    # For quantile loss, we must expand the decoder input to match dec_in (which is 21)
                    num_quantiles = len(self.args.quantiles)  # e.g., 3

                    # 1. Create the "zero" part with the correct quantile dimension
                    # Use dec_in here, which you set to 21 in the script
                    zeros_shape = (batch_y.shape[0], self.args.pred_len, self.args.dec_in)
                    dec_inp_zeros = torch.zeros(zeros_shape).float().to(self.device)

                    # 2. Create the "label" part by repeating the ground truth for each quantile
                    label_part = batch_y[:, :self.args.label_len, :]  # Shape: (batch, label_len, 7)

                    # Repeat tensor 'num_quantiles' times along the last dimension
                    # Shape becomes (batch, label_len, 7 * 3) = (batch, label_len, 21)
                    dec_inp_label = label_part.repeat(1, 1, num_quantiles)

                    # 3. Concatenate them
                    dec_inp = torch.cat([dec_inp_label, dec_inp_zeros], dim=1).float().to(self.device)

                else:
                    # Original logic
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
                # print(outputs.shape,batch_y.shape)
                if self.args.loss == 'quantile' and self.args.features == 'MS':
                    # We need the quantiles for the *target* feature only.
                    # Since target is the last feature, we take the last num_quantiles channels.
                    num_quantiles = len(self.args.quantiles)
                    outputs = outputs[:, -self.args.pred_len:, -num_quantiles:]  # Shape (batch, pred_len, 3)
                else:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                # NEW CODE for test() method
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    # Get ground truth for the last feature (e.g., incidenza)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    seq_len = self.args.seq_len

                    # --- Reverscaling Helper Logic ---
                    target_scale_val = None
                    target_min_or_mean = None
                    real_sigma = 0
                    scaler_type = None

                    if self.global_sigma is None:
                        print("Warning: self.global_sigma not calculated. Run training first (is_training=1).")
                        self.global_sigma = 0

                    if test_data.scale:
                        try:
                            if hasattr(test_data.scaler, 'min_'):  # MinMaxScaler
                                scaler_type = 'minmax'
                                target_scale_val = test_data.scaler.scale_[-1]
                                target_min_or_mean = test_data.scaler.min_[-1]
                                real_sigma = self.global_sigma / target_scale_val
                                gt = (gt / target_scale_val) + target_min_or_mean

                            elif hasattr(test_data.scaler, 'mean_'):  # StandardScaler
                                scaler_type = 'standard'
                                target_scale_val = test_data.scaler.scale_[-1]  # std
                                target_min_or_mean = test_data.scaler.mean_[-1]  # mean
                                real_sigma = self.global_sigma * target_scale_val
                                gt = (gt * target_scale_val) + target_min_or_mean
                        except Exception as e:
                            print(f"Plotting: Could not get scaler params. Error: {e}")
                    else:
                        real_sigma = self.global_sigma
                    # --- END OF REVERSCALING LOGIC ---

                    if self.args.loss == 'quantile':
                        # --- Quantile Plotting Logic ---
                        # ... (This logic is fine, but you should run with MSE) ...
                        # ... (Make sure to reverscale pd_lower_full etc. if you use this) ...
                        print("Error: Plotting in quantile mode. Please use --loss mse to test Method 1.")

                    else:
                        # --- Standard (Non-Quantile) Logic ---
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                        # --- Reverscale Standard Plot ---
                        if test_data.scale and scaler_type:
                            if scaler_type == 'minmax':
                                pd = (pd / target_scale_val) + target_min_or_mean
                            elif scaler_type == 'standard':
                                pd = (pd * target_scale_val) + target_min_or_mean

                        # --- Create Uncertainty Bands ---
                        lower_band = pd - 1.96 * real_sigma
                        upper_band = pd + 1.96 * real_sigma

                        # Clip lower bound at 0
                        lower_band = np.clip(lower_band, a_min=0, a_max=None)

                        visual(true=gt,
                               preds=pd,
                               path=os.path.join(folder_path, str(i) + '.pdf'),
                               lower=lower_band,
                               upper=upper_band,
                               seq_len=seq_len)
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # --- NEW BLOCK TO REVERSCALE METRICS ---
        if test_data.scale:
            print("Reverscaling preds and trues for metrics calculation...")
            try:
                if hasattr(test_data.scaler, 'min_'):  # MinMaxScaler
                    target_scale = test_data.scaler.scale_[-1]
                    target_min = test_data.scaler.min_[-1]
                    preds = (preds / target_scale) + target_min
                    trues = (trues / target_scale) + target_min
                elif hasattr(test_data.scaler, 'mean_'):  # StandardScaler
                    target_scale = test_data.scaler.scale_[-1]  # std
                    target_mean = test_data.scaler.mean_[-1]  # mean
                    preds = (preds * target_scale) + target_mean
                    trues = (trues * target_scale) + target_mean
            except Exception as e:
                print(f"Warning: could not reverscale metrics. Error: {e}")
        else:
            print("Metrics are already on original data scale (no scaling was applied).")
        # --- END OF NEW BLOCK ---

        # mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))
        from utils.metrics import metric

        if self.args.loss == 'quantile':
            results = metric(preds, trues, quantiles=self.args.quantiles)
        else:
            results = metric(preds, trues)

        print(results)

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        # f.write('\n')
        # f.write('\n')
        if isinstance(results, dict):
            for k, v in results.items():
                f.write(f"{k}: {v:.6f}  ")
        else:
            # backward compatibility if results is a tuple
            mae, mse, rmse, mape, mspe, rse, corr = results
            f.write(f"mse:{mse:.6f}, mae:{mae:.6f}, rse:{rse:.6f}, corr:{corr:.6f}")

        f.write('\n\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
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