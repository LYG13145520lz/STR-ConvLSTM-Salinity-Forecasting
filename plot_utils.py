import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
import os
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_regional_salinity_map(csv_path="/spatiotemporal_features.csv"):
    try:
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        target_date = "2020-01-01"
        df_day = df[df["time"].dt.date.astype(str) == target_date]
        df_day_surface = df_day[df_day["depth"] == df_day["depth"].min()]
        lon = df_day_surface["longitude"].values
        lat = df_day_surface["latitude"].values
        salinity = df_day_surface["so_glor"].values

        if len(salinity) == 0:
            raise ValueError(f"No data found for date {target_date}.")

        data_min_lon, data_max_lon = lon.min(), lon.max()
        data_min_lat, data_max_lat = lat.min(), lat.max()
        grid_lon = np.linspace(data_min_lon, data_max_lon, 200)
        grid_lat = np.linspace(data_min_lat, data_max_lat, 200)
        GridLon, GridLat = np.meshgrid(grid_lon, grid_lat)
        grid_salinity = griddata((lon, lat), salinity, (GridLon, GridLat), method="cubic")

        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)

        ax_left = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        ax_right = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())

        regional_extent = [119, 128, 28, 35]

        ax_left.set_extent(regional_extent, crs=ccrs.PlateCarree())
        ax_left.stock_img()
        cf_left = ax_left.contourf(GridLon, GridLat, grid_salinity,
                                   levels=128, cmap='cividis',
                                   transform=ccrs.PlateCarree(),
                                   alpha=0.8, zorder=1)

        land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                                    edgecolor='black', facecolor='gainsboro')
        ax_left.add_feature(land_feature, zorder=2)
        ax_left.add_feature(cfeature.RIVERS, zorder=3)

        rect_left = mpatches.Rectangle((data_min_lon, data_min_lat),
                                       data_max_lon - data_min_lon,
                                       data_max_lat - data_min_lat,
                                       fill=False, edgecolor='#FF4500', linewidth=2,
                                       transform=ccrs.PlateCarree(), zorder=4)
        ax_left.add_patch(rect_left)

        gl_left = ax_left.gridlines(draw_labels=True, linewidth=1, color='white', alpha=0.5, linestyle='--')
        gl_left.top_labels = False
        gl_left.right_labels = False
        gl_left.xformatter = LongitudeFormatter(zero_direction_label=True, degree_symbol='°')
        gl_left.yformatter = LatitudeFormatter(degree_symbol='°')
        gl_left.xlabel_style = {'size': 11}
        gl_left.ylabel_style = {'size': 11}

        ax_right.set_extent(regional_extent, crs=ccrs.PlateCarree())
        ax_right.add_feature(cfeature.OCEAN, facecolor='#eaf6f9', zorder=0)
        cf_right = ax_right.contourf(GridLon, GridLat, grid_salinity,
                                     levels=64, cmap='YlGnBu',
                                     transform=ccrs.PlateCarree(),
                                     zorder=1)
        ax_right.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=2)
        ax_right.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.2, zorder=3)
        ax_right.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', zorder=3)
        ax_right.add_feature(cfeature.RIVERS, zorder=3)

        rect_right = mpatches.Rectangle((data_min_lon, data_min_lat),
                                        data_max_lon - data_min_lon,
                                        data_max_lat - data_min_lat,
                                        fill=False, edgecolor='#FF4500', linewidth=2,
                                        transform=ccrs.PlateCarree(), zorder=4)
        ax_right.add_patch(rect_right)

        gl_right = ax_right.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.3, linestyle='--')
        gl_right.top_labels = False
        gl_right.right_labels = False
        gl_right.xformatter = LongitudeFormatter(zero_direction_label=True, degree_symbol='°')
        gl_right.yformatter = LatitudeFormatter(degree_symbol='°')
        gl_right.xlabel_style = {'size': 11}
        gl_right.ylabel_style = {'size': 11}

        divider = make_axes_locatable(ax_right)
        cax = divider.append_axes("right", size="4%", pad=0.1, axes_class=plt.Axes)
        cbar = fig.colorbar(cf_right, cax=cax)
        cbar.set_label('Salinity (PSU)', fontsize=12)

        plt.savefig("yangtze_dual_style_comparison.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.show()

    except FileNotFoundError:
        print(f"[Error] File not found: {csv_path}")
    except ValueError as ve:
        print(f"[Error] {ve}")
    except Exception as e:
        print(f"[Error] Unexpected error: {e}")

def plot_timeseries_comparison(pkl_path='/results/ConvLSTM_Seq2Seq_predictions_h5.pkl'):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {pkl_path}")

    y_true_final = data['y_true']
    y_pred_final = data['y_pred']
    unique_lats = data['unique_lats']
    unique_lons = data['unique_lons']

    N_steps = 100
    n_samples_available = y_true_final.shape[0]
    if n_samples_available < N_steps:
        N_steps = n_samples_available

    y_true_ts = y_true_final[:N_steps, 0, :, :]
    y_pred_ts = y_pred_final[:N_steps, 0, :, :]

    lats = np.array(unique_lats)
    lons = np.array(unique_lons)

    targets = [
        {"lat": 30.25, "lon": 121.75},
        {"lat": 31.00, "lon": 121.75}
    ]

    indices = []
    for t in targets:
        lat_idx = np.argmin(np.abs(lats - t["lat"]))
        lon_idx = np.argmin(np.abs(lons - t["lon"]))
        indices.append((lat_idx, lon_idx))


    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.linewidth'] = 1.5 

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=150)


    color_obs = '#4A6A95'  
    color_pred = '#E35342' 

    for i, (lat_idx, lon_idx) in enumerate(indices):
        ax = axes[i]
        target_lat = lats[lat_idx]
        target_lon = lons[lon_idx]
        
        obs_data = y_true_ts[:, lat_idx, lon_idx]
        pred_data = y_pred_ts[:, lat_idx, lon_idx]
        

        ax.plot(obs_data, color=color_obs, linewidth=3.0, label='Observed', zorder=3)
        ax.plot(pred_data, color=color_pred, linestyle='--', linewidth=3.0, label='Predicted', zorder=3)
        

        ax.set_title(f"Lat: {target_lat:.2f}°N, Lon: {target_lon:.2f}°E", fontsize=18, fontweight='bold', pad=15)
        ax.set_ylabel("Salinity (PSU)", fontsize=16, fontweight='bold')
        
   
        ax.legend(loc='upper right', fontsize=15, frameon=False, ncol=2)
        
 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#B0B0B0', zorder=0, linewidth=1.2)
        ax.xaxis.grid(True, linestyle=':', alpha=0.5, color='#C0C0C0', zorder=0, linewidth=1.0)
        
        ax.tick_params(axis='both', labelsize=15, width=1.5, length=6)
        

        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))


        ax.autoscale(enable=True, axis='y', tight=True)
        ymin, ymax = ax.get_ylim()
        range_y = ymax - ymin
        ax.set_ylim(ymin - 0.05 * range_y, ymax + 0.15 * range_y)


    axes[1].set_xlabel("Time Step (Day)", fontsize=16, fontweight='bold')

    plt.tight_layout(pad=2.0)
    plt.savefig('Timeseries_Two_Points.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_spatial_variation_analysis(pkl_path='/results/ConvLSTM_Seq2Seq_predictions_h5.pkl'):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        if os.path.exists('ConvLSTM_Seq2Seq_predictions_h5.pkl'):
            with open('ConvLSTM_Seq2Seq_predictions_h5.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            raise FileNotFoundError("Please check the pkl file path.")

    target_val = 2.828
    diffs = []
    for i in range(len(data['y_true_res'])):
        val = np.max(np.abs(data['y_true_res'][i, 4, :, :]))
        diffs.append(abs(val - target_val))

    sample_idx = np.argmin(diffs)
    
    lats = np.array(data['unique_lats'])
    lons = np.array(data['unique_lons'])

    true_raw = data['y_true_res'][sample_idx, 4, :, :]
    pred_raw = data['y_pred_res'][sample_idx, 4, :, :]
    error_raw = true_raw - pred_raw

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    values_true = true_raw.ravel()
    values_pred = pred_raw.ravel()
    values_err = error_raw.ravel()

    new_lons = np.linspace(lons.min(), lons.max(), 100)
    new_lats = np.linspace(lats.min(), lats.max(), 100)
    new_lon_grid, new_lat_grid = np.meshgrid(new_lons, new_lats)
    new_points = np.column_stack([new_lon_grid.ravel(), new_lat_grid.ravel()])

    try:
        true_smooth = griddata(points, values_true, new_points, method='cubic').reshape(len(new_lats), len(new_lons))
        pred_smooth = griddata(points, values_pred, new_points, method='cubic').reshape(len(new_lats), len(new_lons))
        error_smooth = griddata(points, values_err, new_points, method='cubic').reshape(len(new_lats), len(new_lons))
    except Exception:
        true_smooth = griddata(points, values_true, new_points, method='linear').reshape(len(new_lats), len(new_lons))
        pred_smooth = griddata(points, values_pred, new_points, method='linear').reshape(len(new_lats), len(new_lons))
        error_smooth = griddata(points, values_err, new_points, method='linear').reshape(len(new_lats), len(new_lons))

    true_smooth = np.nan_to_num(true_smooth, nan=0.0)
    pred_smooth = np.nan_to_num(pred_smooth, nan=0.0)
    error_smooth = np.nan_to_num(error_smooth, nan=0.0)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    fig, axes = plt.subplots(1, 3, figsize=(26, 8), dpi=300)

    cmap = 'RdBu_r'
    levels_v = np.linspace(-2.828, 2.828, 21)
    levels_e = np.linspace(-1.263, 1.263, 21)

    c1 = axes[0].contourf(new_lons, new_lats, true_smooth, levels=levels_v, cmap=cmap, extend='both')
    axes[0].set_title("(a) True Variation", fontweight='bold', fontsize=22, pad=20)
    axes[0].set_ylabel("Latitude (°N)", fontsize=18)

    c2 = axes[1].contourf(new_lons, new_lats, pred_smooth, levels=levels_v, cmap=cmap, extend='both')
    axes[1].set_title("(b) Predicted Variation", fontweight='bold', fontsize=22, pad=20)

    c3 = axes[2].contourf(new_lons, new_lats, error_smooth, levels=levels_e, cmap=cmap, extend='both')
    axes[2].set_title("(c) Prediction Error", fontweight='bold', fontsize=22, pad=20)

    for i, ax in enumerate(axes):
        ax.set_xlabel("Longitude (°E)", fontsize=18)
        ax.tick_params(labelsize=15)
        ax.set_xticks([121.0, 121.5, 122.0, 122.5, 123.0])
        ax.set_yticks([29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0])
        
        if i < 2:
            cb = fig.colorbar(c1, ax=ax, orientation='vertical', fraction=0.046, pad=0.06)
            cb.set_label('Variation (PSU)', fontsize=16, fontweight='bold')
        else:
            cb = fig.colorbar(c3, ax=ax, orientation='vertical', fraction=0.046, pad=0.06)
            cb.set_label('Error (PSU)', fontsize=16, fontweight='bold')
        cb.ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig('Figure6.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_regional_salinity_map()
    plot_timeseries_comparison()
    plot_spatial_variation_analysis()
