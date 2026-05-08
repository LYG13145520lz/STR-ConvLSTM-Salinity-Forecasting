# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange
import warnings
from scipy.interpolate import griddata
import gc
import os

warnings.filterwarnings('ignore')

def configure_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU enabled: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not detected. Using CPU.")
    return device

def reshape_to_grid_for_prediction(df, features, target, seq_len=7, horizon=3, use_residual=True):
    df['time'] = pd.to_datetime(df['time'])
    print("Step 1/5: Performing temporal linear interpolation...")
    all_cols_to_interpolate = list(dict.fromkeys(features + [target]))
    cols_to_process = [col for col in all_cols_to_interpolate if col in df.columns]
    df[cols_to_process] = df.groupby(['latitude', 'longitude'])[cols_to_process].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    
    print("Step 2/5: Getting grid coordinates...")
    unique_lats, unique_lons, unique_times = sorted(df['latitude'].unique()), sorted(df['longitude'].unique()), sorted(df['time'].unique())
    
    n_lat, n_lon, n_time, n_features = len(unique_lats), len(unique_lons), len(unique_times), len(features)
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
    
    grid_data = np.full((n_time, n_lat, n_lon, n_features), np.nan)
    grid_target = np.full((n_time, n_lat, n_lon), np.nan)
    
    df_indexed = df.set_index('time')
    print("Step 3/5: Initial grid filling...")
    for t_idx, time in enumerate(unique_times):
        time_slice = df_indexed.loc[[time]]
        for _, row in time_slice.iterrows():
            if row['latitude'] in lat_to_idx and row['longitude'] in lon_to_idx:
                lat_idx, lon_idx = lat_to_idx[row['latitude']], lon_to_idx[row['longitude']]
                grid_data[t_idx, lat_idx, lon_idx, :] = row[features].values
                grid_target[t_idx, lat_idx, lon_idx] = row[target]
    
    print("Step 4/5: Spatial interpolation...")
    grid_lons, grid_lats = np.meshgrid(unique_lons, unique_lats)
    for t_idx in range(n_time):
        for f_idx in range(n_features):
            data_slice = grid_data[t_idx, :, :, f_idx]
            if np.isnan(data_slice).any():
                known_points = np.where(~np.isnan(data_slice))
                known_values = data_slice[known_points]
                if len(known_values) > 2:
                    unknown_points = np.where(np.isnan(data_slice))
                    interp_values = griddata(
                        (grid_lats[known_points], grid_lons[known_points]),
                        known_values,
                        (grid_lats[unknown_points], grid_lons[unknown_points]),
                        method='cubic'
                    )
                    data_slice[unknown_points] = interp_values
                    grid_data[t_idx, :, :, f_idx] = data_slice
        
        target_slice = grid_target[t_idx, :, :]
        if np.isnan(target_slice).any():
            known_points = np.where(~np.isnan(target_slice))
            known_values = target_slice[known_points]
            if len(known_values) > 2:
                unknown_points = np.where(np.isnan(target_slice))
                interp_values = griddata(
                    (grid_lats[known_points], grid_lons[known_points]),
                    known_values,
                    (grid_lats[unknown_points], grid_lons[unknown_points]),
                    method='cubic'
                )
                target_slice[unknown_points] = interp_values
                grid_target[t_idx, :, :] = target_slice

    print("Step 5/5: Final cleanup...")
    for t in range(1, n_time):
        mask_data = np.isnan(grid_data[t])
        mask_target = np.isnan(grid_target[t])
        grid_data[t][mask_data] = grid_data[t-1][mask_data]
        grid_target[t][mask_target] = grid_target[t-1][mask_target]

    for f_idx in range(n_features):
        feature_mean = np.nanmean(grid_data[:, :, :, f_idx])
        grid_data[:, :, :, f_idx] = np.nan_to_num(grid_data[:, :, :, f_idx], nan=feature_mean)
    target_mean = np.nanmean(grid_target)
    grid_target = np.nan_to_num(grid_target, nan=target_mean)

    print(f"Building sequences for {'RESIDUAL' if use_residual else 'ABSOLUTE'} prediction...")
    X, y_multi, y_persistence = [], [], []
    for t in range(n_time - seq_len - horizon):
        X.append(grid_data[t : t + seq_len])
        if use_residual:
            residuals = [grid_target[t + seq_len + h] - grid_target[t + seq_len + h - 1] for h in range(horizon)]
            y_multi.append(np.stack(residuals, axis=0))
        else:
            absolutes = [grid_target[t + seq_len + h] for h in range(horizon)]
            y_multi.append(np.stack(absolutes, axis=0))
        y_persistence.append(grid_target[t + seq_len - 1])
    
    return np.array(X), np.array(y_multi), np.array(y_persistence), (unique_lats, unique_lons)

def sliding_window_truncate(X, y, y_persistence, window_size=30, stride=10):
    print(f"Applying Sliding Window Truncation (Window: {window_size}, Stride: {stride})...")
    X_truncated, y_truncated, y_pers_truncated = [], [], []
    total_samples = X.shape[0]
    for start_idx in range(0, total_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        X_truncated.append(X[start_idx:end_idx])
        y_truncated.append(y[start_idx:end_idx])
        y_pers_truncated.append(y_persistence[start_idx:end_idx])
    return (
        np.concatenate(X_truncated, axis=0),
        np.concatenate(y_truncated, axis=0),
        np.concatenate(y_pers_truncated, axis=0)
    )

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim, self.hidden_dim, self.kernel_size = input_dim, hidden_dim, kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=4 * hidden_dim, kernel_size=self.kernel_size, padding=self.padding)
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )
    
class ConvLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, kernel_size, num_layers):
        super(ConvLSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim_list
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim_list[i-1]
            self.layers.append(ConvLSTMCell(cur_input_dim, hidden_dim_list[i], kernel_size))
            self.layers.append(nn.LayerNorm([hidden_dim_list[i]]))

    def forward(self, input_tensor):
        b, seq_len, _, h, w = input_tensor.size()
        hidden_states = self._init_hidden(b, (h, w))
        current_input = input_tensor
        for layer_idx in range(self.num_layers):
            h_c = hidden_states[layer_idx]
            h_t, c_t = h_c
            output_inner = []
            cell = self.layers[layer_idx * 2]
            norm = self.layers[layer_idx * 2 + 1]
            for t in range(seq_len):
                h_t, c_t = cell(input_tensor=current_input[:, t, :, :, :], cur_state=[h_t, c_t])
                h_t_reshaped = h_t.permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim[layer_idx])
                h_t_norm = norm(h_t_reshaped).reshape(b, h, w, self.hidden_dim[layer_idx]).permute(0, 3, 1, 2)
                output_inner.append(h_t_norm)
            layer_output = torch.stack(output_inner, dim=1)
            current_input = layer_output
        return h_t, c_t

    def _init_hidden(self, batch_size, image_size):
        return [self.layers[i*2].init_hidden(batch_size, image_size) for i in range(self.num_layers)]

class ConvLSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim_list, kernel_size, num_layers, horizon):
        super(ConvLSTMDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim_list
        self.horizon = horizon
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = self.hidden_dim[-1] if i == 0 else self.hidden_dim[i]
            self.layers.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], kernel_size))
            self.layers.append(nn.LayerNorm([self.hidden_dim[i]]))
        self.conv_out = nn.Conv2d(self.hidden_dim[-1], output_dim, kernel_size=1)
    def forward(self, encoder_h, encoder_c):
        b, _, h, w = encoder_h.size()
        outputs = []
        # 修复：使用 encoder_h.device 替代未定义的 device 变量
        h_t = [encoder_h] + [torch.zeros_like(encoder_h) for _ in range(self.num_layers - 1)]
        c_t = [encoder_c] + [torch.zeros_like(encoder_c) for _ in range(self.num_layers - 1)]
        decoder_input = torch.zeros(b, self.hidden_dim[-1], h, w, device=encoder_h.device)
        for _ in range(self.horizon):
            current_input = decoder_input
            for layer_idx in range(self.num_layers):
                cell = self.layers[layer_idx * 2]
                norm = self.layers[layer_idx * 2 + 1]
                h_t[layer_idx], c_t[layer_idx] = cell(current_input, [h_t[layer_idx], c_t[layer_idx]])
                h_reshaped = h_t[layer_idx].permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim[layer_idx])
                h_norm = norm(h_reshaped).reshape(b, h, w, self.hidden_dim[layer_idx]).permute(0, 3, 1, 2)
                current_input = h_norm
            output = self.conv_out(current_input)
            outputs.append(output)
        return torch.cat(outputs, dim=1)

class ConvLSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, horizon):
        super(ConvLSTMSeq2Seq, self).__init__()
        self.encoder = ConvLSTMEncoder(input_dim, hidden_dim, kernel_size, num_layers)
        self.decoder = ConvLSTMDecoder(1, hidden_dim, kernel_size, num_layers, horizon)
    def forward(self, input_tensor):
        encoder_h, encoder_c = self.encoder(input_tensor)
        return self.decoder(encoder_h, encoder_c)

class SpatioTemporalDataset(Dataset):
    def __init__(self, X, y, y_persistence):
        self.X = torch.FloatTensor(np.transpose(X, (0, 1, 4, 2, 3)))
        self.y = torch.FloatTensor(y)
        self.y_persistence = torch.FloatTensor(y_persistence)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.y_persistence[idx]

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, target_scaler, device, config, horizon, is_conv_model, use_residual=True):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
    best_val_loss = float('inf')
    
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")
    
    print(f"Training {model_name}...")
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for i, (X_batch, y_batch, _) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if not is_conv_model:
                b, s, c, h, w = X_batch.shape
                X_batch = X_batch.permute(0, 3, 4, 1, 2).reshape(b * h * w, s, c)
                y_batch = y_batch.permute(0, 2, 3, 1).reshape(b * h * w, horizon)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if not is_conv_model:
                    b, s, c, h, w = X_batch.shape
                    X_batch = X_batch.permute(0, 3, 4, 1, 2).reshape(b * h * w, s, c)
                    y_batch = y_batch.permute(0, 2, 3, 1).reshape(b * h * w, horizon)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_{safe_model_name}_h{horizon}.pth')
    
    load_path = f'best_{safe_model_name}_h{horizon}.pth'
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
        print(f"Loaded best model from {load_path}")

    model.eval()
    
    y_true_final_list, y_pred_final_list = [], []
    with torch.no_grad():
        for X_batch, y_batch, y_pers in test_loader:
            X_input = X_batch.to(device)
            b, h, w = y_pers.shape
            if not is_conv_model:
                _, s, c, _, _ = X_input.shape
                X_input_reshaped = X_input.permute(0, 3, 4, 1, 2).reshape(b * h * w, s, c)
                pred_scaled = model(X_input_reshaped).cpu().numpy().reshape(b, h, w, horizon).transpose(0, 3, 1, 2)
            else:
                pred_scaled = model(X_input).cpu().numpy()
            
            if use_residual:
                pred_unscaled = np.stack([
                    target_scaler.inverse_transform(pred_scaled[:, i].reshape(-1, 1)).reshape(b, h, w)
                    for i in range(horizon)
                ], axis=1)
                true_unscaled = np.stack([
                    target_scaler.inverse_transform(y_batch[:, i].numpy().reshape(-1, 1)).reshape(b, h, w)
                    for i in range(horizon)
                ], axis=1)
                pred_final = np.zeros_like(pred_unscaled)
                true_final = np.zeros_like(true_unscaled)
                for i in range(horizon):
                    base_pred = pred_final[:, i-1] if i > 0 else y_pers.numpy()
                    base_true = true_final[:, i-1] if i > 0 else y_pers.numpy()
                    pred_final[:, i] = base_pred + pred_unscaled[:, i]
                    true_final[:, i] = base_true + true_unscaled[:, i]
            else:
                pred_final = np.stack([
                    target_scaler.inverse_transform(pred_scaled[:, i].reshape(-1, 1)).reshape(b, h, w)
                    for i in range(horizon)
                ], axis=1)
                true_final = np.stack([
                    target_scaler.inverse_transform(y_batch[:, i].numpy().reshape(-1, 1)).reshape(b, h, w)
                    for i in range(horizon)
                ], axis=1)
            
            y_pred_final_list.append(pred_final)
            y_true_final_list.append(true_final)
    
    y_true_final = np.concatenate(y_true_final_list, axis=0)
    y_pred_final = np.concatenate(y_pred_final_list, axis=0)
    mae = mean_absolute_error(y_true_final.flatten(), y_pred_final.flatten())
    mse = mean_squared_error(y_true_final.flatten(), y_pred_final.flatten())
    rmse = np.sqrt(mse)
    
    print(f"{model_name} | MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    return {'MAE': mae, 'RMSE': rmse}

def run_ablation_study(file_path="./spatiotemporal_features.csv", horizon=5):
    config = {
        "file_path": file_path,
        "seq_len": 7,
        "batch_size": 16,
        "epochs": 40,
        "lr": 5e-5,
        "test_ratio": 0.2,
        "val_ratio": 0.15,
        "convlstm_hidden_dim": [64, 64],
        "convlstm_kernel_size": (5, 5),
        "convlstm_num_layers": 2,
        "window_size": 50,
        "stride": 25,
        "horizon": horizon
    }

    device = configure_gpu()
    
    print("Starting ablation study for STR-ConvLSTM...")
    df = pd.read_csv(config["file_path"])
    df['time'] = pd.to_datetime(df['time'])
    df = df[df['time'] >= '2008-01-01'].copy()
    print(f"Data loaded: {df.shape[0]} rows, using data from 2008 onwards.")

    full_features = ['uo_glor', 'vo_glor', 'so_glor', 'thetao_glor', 'zos_glor', 'mlotst_glor', 'month_sin', 'month_cos', 'day_sin']
    salinity_only_features = ['so_glor', 'month_sin', 'month_cos', 'day_sin']
    target = 'so_glor'

    all_results = {}

    print("Running Full STR-ConvLSTM (Residual + Multivariate)")
    X_full, y_full, y_pers_full, (lats, lons) = reshape_to_grid_for_prediction(
        df, full_features, target, seq_len=config["seq_len"], horizon=config["horizon"], use_residual=True
    )
    X_full, y_full, y_pers_full = sliding_window_truncate(X_full, y_full, y_pers_full, config["window_size"], config["stride"])

    total = X_full.shape[0]
    test_size = int(total * config["test_ratio"])
    val_size = int((total - test_size) * config["val_ratio"])
    train_size = total - test_size - val_size

    X_train, y_train, y_pers_train = X_full[:train_size], y_full[:train_size], y_pers_full[:train_size]
    X_val, y_val, y_pers_val = X_full[train_size:train_size+val_size], y_full[train_size:train_size+val_size], y_pers_full[train_size:train_size+val_size]
    X_test, y_test, y_pers_test = X_full[train_size+val_size:], y_full[train_size+val_size:], y_pers_full[train_size+val_size:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(X_train.reshape(-1, X_train.shape[-1]))
    scaler_y.fit(y_train.reshape(-1, 1))

    X_train = scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train = scaler_y.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    train_loader = DataLoader(SpatioTemporalDataset(X_train, y_train, y_pers_train), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(SpatioTemporalDataset(X_val, y_val, y_pers_val), batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(SpatioTemporalDataset(X_test, y_test, y_pers_test), batch_size=config["batch_size"], shuffle=False)

    model_full = ConvLSTMSeq2Seq(X_train.shape[-1], config["convlstm_hidden_dim"], config["convlstm_kernel_size"], config["convlstm_num_layers"], config["horizon"])
    res_full = train_and_evaluate(model_full, "STR-ConvLSTM (Full)", train_loader, val_loader, test_loader, scaler_y, device, config, config["horizon"], True, use_residual=True)
    all_results["STR-ConvLSTM (Full)"] = res_full

    del X_full, y_full, y_pers_full, X_train, y_train, y_pers_train, X_val, y_val, y_pers_val, X_test, y_test, y_pers_test
    gc.collect(); torch.cuda.empty_cache()

    print("Running Absolute Variant  (Absolute Prediction)")
    X_abs, y_abs, y_pers_abs, _ = reshape_to_grid_for_prediction(
        df, full_features, target, seq_len=config["seq_len"], horizon=config["horizon"], use_residual=False
    )
    X_abs, y_abs, y_pers_abs = sliding_window_truncate(X_abs, y_abs, y_pers_abs, config["window_size"], config["stride"])

    total = X_abs.shape[0]
    test_size = int(total * config["test_ratio"])
    val_size = int((total - test_size) * config["val_ratio"])
    train_size = total - test_size - val_size

    X_train, y_train, y_pers_train = X_abs[:train_size], y_abs[:train_size], y_pers_abs[:train_size]
    X_val, y_val, y_pers_val = X_abs[train_size:train_size+val_size], y_abs[train_size:train_size+val_size], y_pers_abs[train_size:train_size+val_size]
    X_test, y_test, y_pers_test = X_abs[train_size+val_size:], y_abs[train_size+val_size:], y_pers_abs[train_size+val_size:]

    scaler_X_abs = StandardScaler()
    scaler_y_abs = StandardScaler()
    scaler_X_abs.fit(X_train.reshape(-1, X_train.shape[-1]))
    scaler_y_abs.fit(y_train.reshape(-1, 1))

    X_train = scaler_X_abs.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler_X_abs.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler_X_abs.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train = scaler_y_abs.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val = scaler_y_abs.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test = scaler_y_abs.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    train_loader = DataLoader(SpatioTemporalDataset(X_train, y_train, y_pers_train), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(SpatioTemporalDataset(X_val, y_val, y_pers_val), batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(SpatioTemporalDataset(X_test, y_test, y_pers_test), batch_size=config["batch_size"], shuffle=False)

    model_abs = ConvLSTMSeq2Seq(X_train.shape[-1], config["convlstm_hidden_dim"], config["convlstm_kernel_size"], config["convlstm_num_layers"], config["horizon"])
    res_abs = train_and_evaluate(model_abs, "STR-ConvLSTM w/o Residual", train_loader, val_loader, test_loader, scaler_y_abs, device, config, config["horizon"], True, use_residual=False)
    all_results["STR-ConvLSTM w/o Residual"] = res_abs

    del X_abs, y_abs, y_pers_abs, X_train, y_train, y_pers_train, X_val, y_val, y_pers_val, X_test, y_test, y_pers_test
    gc.collect(); torch.cuda.empty_cache()

    print("Running Univariate Variant (STR-ConvLSTM w/o Physical Covariates)")
    X_sal, y_sal, y_pers_sal, _ = reshape_to_grid_for_prediction(
        df, salinity_only_features, target, seq_len=config["seq_len"], horizon=config["horizon"], use_residual=True
    )
    X_sal, y_sal, y_pers_sal = sliding_window_truncate(X_sal, y_sal, y_pers_sal, config["window_size"], config["stride"])

    total = X_sal.shape[0]
    test_size = int(total * config["test_ratio"])
    val_size = int((total - test_size) * config["val_ratio"])
    train_size = total - test_size - val_size

    X_train, y_train, y_pers_train = X_sal[:train_size], y_sal[:train_size], y_pers_sal[:train_size]
    X_val, y_val, y_pers_val = X_sal[train_size:train_size+val_size], y_sal[train_size:train_size+val_size], y_pers_sal[train_size:train_size+val_size]
    X_test, y_test, y_pers_test = X_sal[train_size+val_size:], y_sal[train_size+val_size:], y_pers_sal[train_size+val_size:]

    scaler_X_sal = StandardScaler()
    scaler_y_sal = StandardScaler()
    scaler_X_sal.fit(X_train.reshape(-1, X_train.shape[-1]))
    scaler_y_sal.fit(y_train.reshape(-1, 1))

    X_train = scaler_X_sal.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler_X_sal.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler_X_sal.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train = scaler_y_sal.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_val = scaler_y_sal.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)
    y_test = scaler_y_sal.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    train_loader = DataLoader(SpatioTemporalDataset(X_train, y_train, y_pers_train), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(SpatioTemporalDataset(X_val, y_val, y_pers_val), batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(SpatioTemporalDataset(X_test, y_test, y_pers_test), batch_size=config["batch_size"], shuffle=False)

    model_sal = ConvLSTMSeq2Seq(X_train.shape[-1], config["convlstm_hidden_dim"], config["convlstm_kernel_size"], config["convlstm_num_layers"], config["horizon"])
    res_sal = train_and_evaluate(model_sal, "STR-ConvLSTM w/o Physical Covariates", train_loader, val_loader, test_loader, scaler_y_sal, device, config, config["horizon"], True, use_residual=True)
    all_results["STR-ConvLSTM w/o Physical Covariates"] = res_sal

    print("\nAblation Study Results")
    results_df = pd.DataFrame(all_results).T
    print(results_df[['MAE', 'RMSE']])

    best_model = results_df['RMSE'].idxmin()
    print(f"Best model: {best_model}")
    
    return all_results

if __name__ == "__main__":
    results = run_ablation_study()
