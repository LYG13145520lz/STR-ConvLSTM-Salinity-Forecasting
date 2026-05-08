# -*- coding: utf-8 -*-
#==============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error # Removed r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange
import warnings
from scipy.interpolate import griddata
import gc
import math
import os
import pickle

warnings.filterwarnings('ignore')

# --------------------------- GPU Configuration ---------------------------
def configure_gpu() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU enabled: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU not detected. Using CPU.")
    return device

device = configure_gpu()

# --------------------------- Data Processing for Multi-Step Residuals ---------------------------
def reshape_to_grid_for_residual_prediction(df: pd.DataFrame, features: list, target: str, seq_len: int = 7, horizon: int = 3) -> tuple:
    df['time'] = pd.to_datetime(df['time'])
    print("Step 1/5: Performing temporal linear interpolation...")
    all_cols_to_interpolate = list(dict.fromkeys(features + [target]))
    cols_to_process = [col for col in all_cols_to_interpolate if col in df.columns]
    df[cols_to_process] = df.groupby(['latitude', 'longitude'])[cols_to_process].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    with tqdm(total=3, desc="Step 2/5: Getting grid coordinates") as pbar:
        unique_lats, unique_lons, unique_times = sorted(df['latitude'].unique()), sorted(df['longitude'].unique()), sorted(df['time'].unique()); pbar.update(3)
    n_lat, n_lon, n_time, n_features = len(unique_lats), len(unique_lons), len(unique_times), len(features)
    lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}; lon_to_idx = {lon: i for i, lon in enumerate(unique_lons)}
    grid_data, grid_target = np.full((n_time, n_lat, n_lon, n_features), np.nan), np.full((n_time, n_lat, n_lon), np.nan)
    df_indexed = df.set_index('time')
    for t_idx, time in enumerate(tqdm(unique_times, desc="Step 3/5: Initial grid filling")):
        time_slice = df_indexed.loc[[time]]
        for _, row in time_slice.iterrows():
            if row['latitude'] in lat_to_idx and row['longitude'] in lon_to_idx:
                lat_idx, lon_idx = lat_to_idx[row['latitude']], lon_to_idx[row['longitude']]
                grid_data[t_idx, lat_idx, lon_idx, :] = row[features].values
                grid_target[t_idx, lat_idx, lon_idx] = row[target]
    grid_lons, grid_lats = np.meshgrid(unique_lons, unique_lats)
    for t_idx in tqdm(range(n_time), desc="Step 4/5: Spatial interpolation"):
        for f_idx in range(n_features):
            data_slice = grid_data[t_idx, :, :, f_idx]
            if np.isnan(data_slice).any():
                known_points, known_values = np.where(~np.isnan(data_slice)), data_slice[np.where(~np.isnan(data_slice))]
                if len(known_values) > 2:
                    unknown_points = np.where(np.isnan(data_slice))
                    interp_values = griddata((grid_lats[known_points], grid_lons[known_points]), known_values, (grid_lats[unknown_points], grid_lons[unknown_points]), method='cubic')
                    data_slice[unknown_points] = interp_values; grid_data[t_idx, :, :, f_idx] = data_slice
        target_slice = grid_target[t_idx, :, :]
        if np.isnan(target_slice).any():
            known_points, known_values = np.where(~np.isnan(target_slice)), target_slice[np.where(~np.isnan(target_slice))]
            if len(known_values) > 2:
                unknown_points = np.where(np.isnan(target_slice))
                interp_values = griddata((grid_lats[known_points], grid_lons[known_points]), known_values, (grid_lats[unknown_points], grid_lons[unknown_points]), method='cubic')
                target_slice[unknown_points] = interp_values; grid_target[t_idx, :, :] = target_slice
    print("Step 5/5: Final cleanup...")
    for t in tqdm(range(1, n_time), desc="...forward fill"):
        mask_data, mask_target = np.isnan(grid_data[t]), np.isnan(grid_target[t])
        grid_data[t][mask_data], grid_target[t][mask_target] = grid_data[t-1][mask_data], grid_target[t-1][mask_target]
    for f_idx in range(n_features):
        feature_mean = np.nanmean(grid_data[:, :, :, f_idx]); grid_data[:, :, :, f_idx] = np.nan_to_num(grid_data[:, :, :, f_idx], nan=feature_mean)
    target_mean = np.nanmean(grid_target); grid_target = np.nan_to_num(grid_target, nan=target_mean)
    print(f"\n[CRITICAL] Building sequences for {horizon}-step RESIDUAL prediction...")
    X, y_residual_multi, y_persistence = [], [], []
    for t in tqdm(range(n_time - seq_len - horizon), desc="Building multi-step residual sequences"):
        X.append(grid_data[t : t + seq_len])
        residuals = [grid_target[t + seq_len + h] - grid_target[t + seq_len + h - 1] for h in range(horizon)]
        y_residual_multi.append(np.stack(residuals, axis=0))
        y_persistence.append(grid_target[t + seq_len - 1])
    return np.array(X), np.array(y_residual_multi), np.array(y_persistence), (unique_lats, unique_lons)

def sliding_window_truncate(X, y_residual, y_persistence, window_size=30, stride=10):
    print(f"\n[INFO] Applying Sliding Window Truncation (Window: {window_size}, Stride: {stride})...")
    X_truncated, y_res_truncated, y_pers_truncated = [], [], []
    total_samples = X.shape[0]
    for start_idx in trange(0, total_samples - window_size + 1, stride, desc="Truncating"):
        end_idx = start_idx + window_size
        X_truncated.append(X[start_idx:end_idx]); y_res_truncated.append(y_residual[start_idx:end_idx]); y_pers_truncated.append(y_persistence[start_idx:end_idx])
    return (np.concatenate(X_truncated, axis=0), np.concatenate(y_res_truncated, axis=0), np.concatenate(y_pers_truncated, axis=0))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__(); self.input_dim, self.hidden_dim, self.kernel_size = input_dim, hidden_dim, kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=4 * hidden_dim, kernel_size=self.kernel_size, padding=self.padding)
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1); combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g; h_next = o * torch.tanh(c_next)
        return h_next, c_next
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device), torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, kernel_size, num_layers):
        super(ConvLSTMEncoder, self).__init__(); self.num_layers, self.hidden_dim = num_layers, hidden_dim_list; self.layers = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i-1]
            self.layers.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], kernel_size)); self.layers.append(nn.LayerNorm([self.hidden_dim[i]]))
    def forward(self, input_tensor):
        b, seq_len, _, h, w = input_tensor.size(); hidden_states = self._init_hidden(b, (h, w)); current_input = input_tensor
        for layer_idx in range(self.num_layers):
            h_c = hidden_states[layer_idx]; h_t, c_t = h_c; output_inner = []; cell, norm = self.layers[layer_idx * 2], self.layers[layer_idx * 2 + 1]
            for t in range(seq_len):
                h_t, c_t = cell(input_tensor=current_input[:, t, :, :, :], cur_state=[h_t, c_t])
                h_t_reshaped = h_t.permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim[layer_idx])
                h_t_norm = norm(h_t_reshaped).reshape(b, h, w, self.hidden_dim[layer_idx]).permute(0, 3, 1, 2)
                output_inner.append(h_t_norm)
            layer_output = torch.stack(output_inner, dim=1); current_input = layer_output
        return h_t, c_t
    def _init_hidden(self, batch_size, image_size):
        return [self.layers[i*2].init_hidden(batch_size, image_size) for i in range(self.num_layers)]

class ConvLSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim_list, kernel_size, num_layers, horizon):
        super(ConvLSTMDecoder, self).__init__(); self.num_layers, self.hidden_dim, self.horizon = num_layers, hidden_dim_list, horizon; self.layers = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = self.hidden_dim[-1] if i == 0 else self.hidden_dim[i]
            self.layers.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], kernel_size)); self.layers.append(nn.LayerNorm([self.hidden_dim[i]]))
        self.conv_out = nn.Conv2d(self.hidden_dim[-1], output_dim, kernel_size=1)
    def forward(self, encoder_h, encoder_c):
        b, _, h, w = encoder_h.size(); outputs = []
        h_t = [encoder_h] + [torch.zeros_like(encoder_h, device=device) for _ in range(self.num_layers - 1)]
        c_t = [encoder_c] + [torch.zeros_like(encoder_c, device=device) for _ in range(self.num_layers - 1)]
        decoder_input = torch.zeros(b, self.hidden_dim[-1], h, w, device=device)
        for _ in range(self.horizon):
            current_input = decoder_input
            for layer_idx in range(self.num_layers):
                cell, norm = self.layers[layer_idx * 2], self.layers[layer_idx * 2 + 1]
                h_t[layer_idx], c_t[layer_idx] = cell(current_input, [h_t[layer_idx], c_t[layer_idx]])
                h_reshaped = h_t[layer_idx].permute(0, 2, 3, 1).reshape(b * h * w, self.hidden_dim[layer_idx])
                h_norm = norm(h_reshaped).reshape(b, h, w, self.hidden_dim[layer_idx]).permute(0, 3, 1, 2)
                current_input = h_norm
            output = self.conv_out(current_input); outputs.append(output)
        return torch.cat(outputs, dim=1)

class ConvLSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, horizon):
        super(ConvLSTMSeq2Seq, self).__init__()
        self.encoder = ConvLSTMEncoder(input_dim, hidden_dim, kernel_size, num_layers)
        self.decoder = ConvLSTMDecoder(1, hidden_dim, kernel_size, num_layers, horizon)
    def forward(self, input_tensor):
        encoder_h, encoder_c = self.encoder(input_tensor); return self.decoder(encoder_h, encoder_c)

# --------------------------- Other Model Definitions ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super(LSTMModel, self).__init__(); self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True); self.fc = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device); c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0)); return self.fc(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super(GRUModel, self).__init__(); self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True); self.fc = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0); return self.fc(out[:, -1, :])
        
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        from torch.nn.utils import weight_norm
        class Chomp1d(nn.Module):
            def __init__(self, chomp_size): super(Chomp1d, self).__init__(); self.chomp_size = chomp_size
            def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()
        class TemporalBlock(nn.Module):
            def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
                super(TemporalBlock, self).__init__(); self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)); self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout); self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)); self.chomp2 = Chomp1d(padding); self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout); self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2); self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None; self.relu = nn.ReLU(); self.init_weights()
            def init_weights(self): self.conv1.weight.data.normal_(0, 0.01); self.conv2.weight.data.normal_(0, 0.01); 
            def forward(self, x): out = self.net(x); res = x if self.downsample is None else self.downsample(x); return self.relu(out + res)
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i; in_channels = input_size if i == 0 else num_channels[i-1]; out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers); self.linear = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        x = x.permute(0, 2, 1); output = self.network(x); return self.linear(output[:, :, -1])

class InformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, horizon):
        super(InformerModel, self).__init__(); self.encoder = nn.Linear(input_dim, model_dim); self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, model_dim)); self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True); self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers); self.decoder = nn.Linear(model_dim, horizon)
    def forward(self, src):
        src = self.encoder(src) + self.pos_encoder[:, :src.size(1), :]; output = self.transformer_encoder(src); return self.decoder(output[:, -1, :])

class AutoformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, horizon):
        super(AutoformerModel, self).__init__(); self.encoder = nn.Linear(input_dim, model_dim); self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, model_dim)); self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True); self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers); self.decoder = nn.Linear(model_dim, horizon); self.seasonal_decomp = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
    def forward(self, x):
        seasonal_output = self.seasonal_decomp(x.permute(0, 2, 1)).permute(0, 2, 1); trend_output = x - seasonal_output
        x = self.encoder(trend_output) + self.pos_encoder[:, :x.size(1), :]; output = self.transformer_encoder(x); return self.decoder(output[:, -1, :])

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, horizon):
        super(TransformerModel, self).__init__(); self.encoder = nn.Linear(input_dim, model_dim); self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, model_dim)); self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True); self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers); self.decoder = nn.Linear(model_dim, horizon)
    def forward(self, src):
        src = self.encoder(src) + self.pos_encoder[:, :src.size(1), :]; output = self.transformer_encoder(src); return self.decoder(output[:, -1, :])

class SpatioTemporalDatasetResidual(Dataset):
    def __init__(self, X, y_residual, y_persistence):
        self.X, self.y_residual, self.y_persistence = torch.FloatTensor(np.transpose(X, (0, 1, 4, 2, 3))), torch.FloatTensor(y_residual), torch.FloatTensor(y_persistence)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_residual[idx], self.y_persistence[idx]

def train_model_residual(model, model_name, train_loader, val_loader, test_loader, target_scaler, device, config, horizon, is_conv_model):
    model.to(device); accumulation_steps = config.get("accumulation_steps", 1); print(f"\nTraining {model_name} for {horizon}-step RESIDUALS...")
    criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=config["lr"]); scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True); best_val_loss = float('inf')
    for epoch in range(config["epochs"]):
        model.train(); train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", leave=False)
        for i, (X_batch, y_batch_res, _) in enumerate(pbar):
            X_batch, y_batch_res = X_batch.to(device), y_batch_res.to(device)
            if not is_conv_model:
                b, s, c, h, w = X_batch.shape; X_batch = X_batch.permute(0, 3, 4, 1, 2).reshape(b * h * w, s, c); y_batch_res = y_batch_res.permute(0, 2, 3, 1).reshape(b * h * w, horizon)
            outputs_res = model(X_batch); loss = criterion(outputs_res, y_batch_res); loss_scaled = loss / accumulation_steps; loss_scaled.backward(); train_loss += loss.item()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader): optimizer.step(); optimizer.zero_grad()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        avg_train_loss = train_loss / len(train_loader); model.eval(); val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch_res, _ in val_loader:
                X_batch, y_batch_res = X_batch.to(device), y_batch_res.to(device)
                if not is_conv_model:
                    b, s, c, h, w = X_batch.shape; X_batch = X_batch.permute(0, 3, 4, 1, 2).reshape(b * h * w, s, c); y_batch_res = y_batch_res.permute(0, 2, 3, 1).reshape(b * h * w, horizon)
                outputs_res = model(X_batch); val_loss += criterion(outputs_res, y_batch_res).item()
        avg_val_loss = val_loss / len(val_loader); scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; torch.save(model.state_dict(), f'best_{model_name}_model_h{horizon}.pth'); print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} -> Saved")
        else: print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    model.load_state_dict(torch.load(f'best_{model_name}_model_h{horizon}.pth')); model.eval(); y_true_final_list, y_pred_final_list, y_true_res_list, y_pred_res_list = [], [], [], []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating {model_name}", leave=False)
        for X_batch, y_batch_res, y_batch_pers in pbar:
            X_input, b, h, w = X_batch.to(device), y_batch_pers.shape[0], y_batch_pers.shape[1], y_batch_pers.shape[2]
            if not is_conv_model:
                _, s, c, _, _ = X_input.shape; X_input_reshaped = X_input.permute(0, 3, 4, 1, 2).reshape(b * h * w, s, c); pred_res_scaled = model(X_input_reshaped).cpu().numpy().reshape(b, h, w, horizon).transpose(0, 3, 1, 2)
            else: pred_res_scaled = model(X_input).cpu().numpy()
            pred_res_unscaled = np.stack([target_scaler.inverse_transform(pred_res_scaled[:, i].reshape(-1, 1)).reshape(b, h, w) for i in range(horizon)], axis=1)
            true_res_unscaled = np.stack([target_scaler.inverse_transform(y_batch_res[:, i].numpy().reshape(-1, 1)).reshape(b, h, w) for i in range(horizon)], axis=1)
            pred_final, true_final = np.zeros_like(pred_res_unscaled), np.zeros_like(true_res_unscaled)
            for i in range(horizon):
                base_pred = pred_final[:, i-1] if i > 0 else y_batch_pers.numpy(); base_true = true_final[:, i-1] if i > 0 else y_batch_pers.numpy()
                pred_final[:, i] = base_pred + pred_res_unscaled[:, i]; true_final[:, i] = base_true + true_res_unscaled[:, i]
            y_pred_res_list.append(pred_res_unscaled); y_true_res_list.append(true_res_unscaled); y_pred_final_list.append(pred_final); y_true_final_list.append(true_final)
    y_true_final, y_pred_final = np.concatenate(y_true_final_list, axis=0), np.concatenate(y_pred_final_list, axis=0)
    y_true_residual, y_pred_residual = np.concatenate(y_true_res_list, axis=0), np.concatenate(y_pred_res_list, axis=0)
    

    mae = mean_absolute_error(y_true_final.flatten(), y_pred_final.flatten())
    mse = mean_squared_error(y_true_final.flatten(), y_pred_final.flatten())
    rmse = np.sqrt(mse)
    
    print(f"\nFinal {model_name} Performance | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
    
    save_dir = "./final_result_h5"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_predictions_h{horizon}.pkl")
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'y_true': y_true_final,
            'y_pred': y_pred_final,
            'y_true_res': y_true_residual,
            'y_pred_res': y_pred_residual,
            'unique_lats': unique_lats,
            'unique_lons': unique_lons,
            'metrics': {'MAE': mae, 'MSE': rmse}
        }, f)
    print(f"✅ Predictions saved to: {save_path}")
   
    return y_true_final, y_pred_final, {'MAE': mae, 'RMSE': rmse}



# --------------------------- Main Execution ---------------------------
config = {
    "file_path": "/root/spatiotemporal_features.csv", "seq_len": 7, "batch_size": 16, "epochs": 40, "lr": 5e-5,
    "test_ratio": 0.2, "val_ratio": 0.15, "rnn_hidden_size": 128, "rnn_num_layers": 2,
    "convlstm_hidden_dim": [64, 64], "convlstm_kernel_size": (5, 5), "convlstm_num_layers": 2,
    "tcn_num_channels": [50, 50, 50], "tcn_kernel_size": 3, "tcn_dropout": 0.2,
    "transformer_model_dim": 128, "transformer_num_heads": 4, "transformer_num_layers": 3,
    "window_size": 50, "stride": 25, 
    "horizon": 5 # To run experiments with different forecast horizons, modify the `horizon` parameter in the `config` dictionary: Options: 1, 3, 5 
}

print("===== MULTI-STEP RESIDUAL PREDICTION EXPERIMENT (WITH OPTIMIZED ConvLSTM) =====")
df = pd.read_csv(config["file_path"]); df['time'] = pd.to_datetime(df['time']); df = df[df['time'] >= '2008-01-01'].copy()
print(f"Data loaded: {df.shape[0]} rows, using data from 2008 onwards.")
features = ['uo_glor', 'vo_glor', 'so_glor', 'thetao_glor', 'zos_glor', 'mlotst_glor', 'month_sin', 'month_cos', 'day_sin']
target = 'so_glor'
X_grid, y_grid_residual, y_grid_persistence, (unique_lats, unique_lons) = reshape_to_grid_for_residual_prediction(df, features, target, seq_len=config["seq_len"], horizon=config["horizon"])
X_grid, y_grid_residual, y_grid_persistence = sliding_window_truncate(X_grid, y_grid_residual, y_grid_persistence, window_size=config["window_size"], stride=config["stride"])

print("\n===== Splitting Datasets =====")
total_samples = X_grid.shape[0]; test_size = int(total_samples * config["test_ratio"]); val_size = int((total_samples-test_size) * config["val_ratio"]); train_size = total_samples - test_size - val_size
X_train, y_res_train, y_pers_train = X_grid[:train_size], y_grid_residual[:train_size], y_grid_persistence[:train_size]
X_val, y_res_val, y_pers_val = X_grid[train_size:train_size+val_size], y_grid_residual[train_size:train_size+val_size], y_grid_persistence[train_size:train_size+val_size]
X_test, y_res_test, y_pers_test = X_grid[train_size+val_size:], y_grid_residual[train_size+val_size:], y_grid_persistence[train_size+val_size:]

print("\n===== Standardization =====")
n_features = X_train.shape[-1]; feature_scaler = StandardScaler(); target_scaler = StandardScaler()
feature_scaler.fit(X_train.reshape(-1, n_features)); target_scaler.fit(y_res_train.reshape(-1, 1))
X_train = feature_scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
X_val = feature_scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
X_test = feature_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
y_res_train = target_scaler.transform(y_res_train.reshape(-1, 1)).reshape(y_res_train.shape)
y_res_val = target_scaler.transform(y_res_val.reshape(-1, 1)).reshape(y_res_val.shape)
y_res_test = target_scaler.transform(y_res_test.reshape(-1, 1)).reshape(y_res_test.shape)

del df, X_grid, y_grid_residual, y_grid_persistence; gc.collect(); torch.cuda.empty_cache()

train_loader = DataLoader(SpatioTemporalDatasetResidual(X_train, y_res_train, y_pers_train), batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(SpatioTemporalDatasetResidual(X_val, y_res_val, y_pers_val), batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(SpatioTemporalDatasetResidual(X_test, y_res_test, y_pers_test), batch_size=config["batch_size"], shuffle=False)

all_results, all_predictions = {}, {}
horizon = config["horizon"]
models_to_run = {
    "ConvLSTM_Seq2Seq": ConvLSTMSeq2Seq(X_train.shape[-1], config["convlstm_hidden_dim"], config["convlstm_kernel_size"], config["convlstm_num_layers"], horizon),
    "LSTM": LSTMModel(X_train.shape[-1], config["rnn_hidden_size"], config["rnn_num_layers"], horizon),
    "GRU": GRUModel(X_train.shape[-1], config["rnn_hidden_size"], config["rnn_num_layers"], horizon),
    "TCN": TCNModel(X_train.shape[-1], horizon, config["tcn_num_channels"], config["tcn_kernel_size"], config["tcn_dropout"]),
    "Informer": InformerModel(X_train.shape[-1], config["transformer_model_dim"], config["transformer_num_heads"], config["transformer_num_layers"], horizon),
    "Autoformer": AutoformerModel(X_train.shape[-1], config["transformer_model_dim"], config["transformer_num_heads"], config["transformer_num_layers"], horizon),
    "Transformer": TransformerModel(X_train.shape[-1], config["transformer_model_dim"], config["transformer_num_heads"], config["transformer_num_layers"], horizon)
}

for model_name, model in models_to_run.items():
    is_conv_model = isinstance(model, ConvLSTMSeq2Seq)
    y_true, y_pred, results = train_model_residual(model, model_name, train_loader, val_loader, test_loader, target_scaler, device, config, horizon, is_conv_model)
    all_results[model_name] = results
    all_predictions[model_name] = (y_true, y_pred)

print("\n\n===== FINAL RESULTS (WITH OPTIMIZED ConvLSTM) =====")
results_df = pd.DataFrame(all_results).T

print(results_df[['MAE', 'RMSE']])

if not results_df.empty:
    best_model = results_df['RMSE'].idxmin()
    print(f"\nBest model based on lowest RMSE: {best_model}")
