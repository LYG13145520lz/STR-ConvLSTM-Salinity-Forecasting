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

