# STR-ConvLSTM-Salinity-Forecasting
# STR-ConvLSTM: Spatiotemporal Residual Learning for Salinity Forecasting

This repository contains the official implementation of the **STR-ConvLSTM** model as described in the manuscript: *"STR-ConvLSTM: A Spatiotemporal Residual Learning Framework for Multivariate Salinity Forecasting in the Yangtze River Estuary"*.

## Key Features
- **Residual Learning Strategy**: Reformulates salinity prediction from absolute values to temporal increments to handle non-stationarity.
- **Multivariate Integration**: Synergistically combines SSS with current velocity, SST, and SSH.
- **Hierarchical Preprocessing**: Robust 4-stage data reconstruction pipeline for estuarine datasets.

## Requirements
- PyTorch 2.0+
- Pandas, NumPy, Scikit-learn
- Scipy (for spatial interpolation)


## 📊 Visualization Tools

### `visualization.py`

This module contains functions for generating publication-quality figures for the STR-ConvLSTM salinity prediction study.

#### 🔧 Functions

| Function | Description | Output |
|----------|-------------|--------|
| `plot_regional_salinity_map()` | Generates dual-style regional salinity maps using Cartopy (texture background + solid color background) | `yangtze_dual_style_comparison.svg` |
| `plot_timeseries_comparison()` | Plots observed vs. predicted salinity time series at two representative grid points | `Timeseries_Two_Points.png` |
| `plot_spatial_variation_analysis()` | Visualizes true variation, predicted variation, and prediction error for a representative sample | `Figure6.png` |

#### 📦 Dependencies

```bash
numpy>=1.21
matplotlib>=3.5
pandas>=1.3
scipy>=1.7
cartopy>=0.20  # for geographic mapping

