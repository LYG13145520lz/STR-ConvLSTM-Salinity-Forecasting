# STR-ConvLSTM-Salinity-Forecasting
# STR-ConvLSTM: Spatiotemporal Residual Learning for Salinity Forecasting

This repository contains the official implementation of the **STR-ConvLSTM** model as described in the manuscript: *"STR-ConvLSTM: A Spatiotemporal Residual Learning Framework for Multivariate Salinity Forecasting in the Yangtze River Estuary"*.

## 📊 Data Availability

### Raw Dataset

The complete spatiotemporal salinity dataset used in this study is publicly available at **Figshare**:

| Item | Details |
|------|---------|
| **Repository** | Figshare |
| **DOI** | `10.6084/m9.figshare.31041805` |
| **Direct Link** | 🔗 [https://doi.org/10.6084/m9.figshare.31041805](https://doi.org/10.6084/m9.figshare.31041805) |
| **License** | CC BY 4.0 |

## Experiment Execution Guide
This section describes how to replicate the full experimental results by running the statistical baselines and the deep learning suite across different forecast horizons. 
Python Version: This environment is based on Python 3.7.
### STR_ConvLSTM_Core.py
### run_baselines.py
1. Prerequisites and Data Preparation
Before running the experiments, ensure you have the complete dataset.
Dataset File: spatiotemporal_features.csv
Location: Place the CSV file in the root directory (or update the file_path in the config dictionary within the scripts).
Time Range: The scripts are pre-configured to filter data from 2008-01-01 onwards to ensure data quality and consistency.
2. Running Statistical Baselines
The run_baselines.py script computes the performance for the Persistence and ARIMA models. These models provide the benchmark for evaluating the necessity of deep learning architectures. 
python run_baselines.py
3. Running Deep Learning Core
The STR_ConvLSTM_Core.py script handles the training and evaluation of 7 different architectures: ConvLSTM, LSTM, GRU, TCN, Informer, Autoformer, and Transformer.
python run_baselines.py
4. Multi-Horizon Analysis
To obtain the full set of results presented in the study, you must conduct experiments across multiple forecast horizons (h)
How to change the Horizon:
In both run_baselines.py and STR_ConvLSTM_Core.py, locate the config dictionary at the beginning of the main execution block: config = {
    ...
    "horizon": 3,  # Modify this value to 1, 3, or 5
    ...
}
5. Result Collection and Evaluation
Console Output: After each run, the scripts will print a summary table containing the MAE and RMSE for every model.
Pickle Files: For the deep learning experiments, the script creates a directory (e.g., ./final_results) and saves .pkl files containing the raw y_true and y_pred arrays.
Naming Convention: Saved files are named as {model_name}_predictions_h{horizon}.pkl to prevent overwriting results from different horizons.

## 📊 Visualization Tools

### `visualization.py`

This module contains functions for generating publication-quality figures for the STR-ConvLSTM salinity prediction study. ```python STR_ConvLSTM_Core.py

#### 🔧 Functions

| Function | Description | Output |
|----------|-------------|--------|
| `plot_regional_salinity_map()` | Generates dual-style regional salinity maps using Cartopy (texture background + solid color background) | `yangtze_dual_style_comparison.svg` |
| `plot_timeseries_comparison()` | Plots observed vs. predicted salinity time series at two representative grid points | `Timeseries_Two_Points.png` |
| `plot_spatial_variation_analysis()` | Visualizes true variation, predicted variation, and prediction error for a representative sample | `Figure6.png` |

#### 📦 Dependencies


numpy>=1.21
matplotlib>=3.5
pandas>=1.3
scipy>=1.7
cartopy>=0.20 

## 🔬 Ablation Study

### `ablation_study.py`

This script runs the ablation study for STR-ConvLSTM, comparing three model variants:

| Variant | Description |
|---------|-------------|
| `STR-ConvLSTM (Full)` | Proposed model with residual learning + multivariate physical covariates |
| `STR-ConvLSTM w/o Residual` | Absolute prediction variant (no residual learning) |
| `STR-ConvLSTM w/o Physical Covariates` | Univariate variant (salinity + time features only) |

#### 📦 Dependencies

bash
# Core
numpy>=1.21
pandas>=1.3
torch>=1.9
scikit-learn>=0.24
scipy>=1.7
tqdm>=4.62

# Optional (for visualization)
matplotlib>=3.5
cartopy>=0.20  

