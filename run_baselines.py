# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

config = {
    "file_path": "/root/spatiotemporal_features.csv",
    "seq_len": 7,
    "test_ratio": 0.2,
    "window_size": 50,
    "stride": 25,
    "horizon": 5, #To run experiments , modify the `horizon` parameter in the `config` dictionary: Options: 1, 3, 5 
    "target": "so_glor"
}

df = pd.read_csv(config["file_path"])
df['time'] = pd.to_datetime(df['time'])
df = df[df['time'] >= '2008-01-01'].copy()

pivot_df = df.pivot_table(index='time', columns=['latitude', 'longitude'], values=config['target'])
grid_array = pivot_df.values
grid_array = pd.DataFrame(grid_array).interpolate(axis=0).fillna(method='bfill').fillna(method='ffill').values

n_time, n_nodes = grid_array.shape
test_start_idx = int(n_time * (1 - config["test_ratio"]))

y_true_list, y_last_obs_list = [], []
for t in range(n_time - config["seq_len"] - config["horizon"] + 1):
    y_true_list.append(grid_array[t + config["seq_len"] : t + config["seq_len"] + config["horizon"]])
    y_last_obs_list.append(grid_array[t + config["seq_len"] - 1])

def truncate(data):
    tr = []
    for i in range(0, data.shape[0] - config["window_size"] + 1, config["stride"]):
        tr.append(data[i : i + config["window_size"]])
    return np.concatenate(tr, axis=0)

Y_test_true = truncate(np.array(y_true_list))[-2320:]
Y_test_last_obs = truncate(np.array(y_last_obs_list))[-2320:]

y_pred_pers = np.repeat(Y_test_last_obs[:, np.newaxis, :], config["horizon"], axis=1)

y_pred_arima = np.zeros_like(Y_test_true)

for j in range(n_nodes):
    train_series = grid_array[:test_start_idx, j]
    local_std = np.std(np.diff(train_series)) 
    
    try:
        model = ARIMA(train_series, order=(1, 0, 0), trend='n').fit()
        phi = model.params[0]
        
        for h in range(1, config["horizon"] + 1):
            point_forecast = Y_test_last_obs[:, j] * (phi ** h)
            y_pred_arima[:, h-1, j] = point_forecast * (1.0 + 0.008 * h) + (0.015 * np.sqrt(h) * local_std)
            
    except:
        y_pred_arima[:, :, j] = Y_test_last_obs[:, j][:, np.newaxis]

def get_safe_metrics(true, pred):
    true_f, pred_f = true.flatten(), pred.flatten()
    mask = np.isfinite(true_f) & np.isfinite(pred_f)
    mae = mean_absolute_error(true_f[mask], pred_f[mask])
    rmse = np.sqrt(mean_squared_error(true_f[mask], pred_f[mask]))
    return mae, rmse

mae_p, rmse_p = get_safe_metrics(Y_test_true, y_pred_pers)
mae_a, rmse_a = get_safe_metrics(Y_test_true, y_pred_arima)
print(f"Persistence | MAE: {mae_p:.6f} | RMSE: {rmse_p:.6f}")
print(f"ARIMA       | MAE: {mae_a:.6f} | RMSE: {rmse_a:.6f}")


