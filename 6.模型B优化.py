import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ----------------- 配置 -----------------
MODEL_A_FILE = "model_a_tke_estimation.csv"  # 模型 A 输出（用于作为参考 / 调参）
OUTPUT_DIR = "model_b_star_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPS = 1e-6

# ----------------- 工具函数 -----------------
def safe_inv(x):
    return 1.0 / (np.where(np.isfinite(x) & (x > 0), x, np.nan) + EPS)

def rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((a[mask] - b[mask])**2))

def mae(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(a[mask] - b[mask]))

# ----------------- 加载数据 -----------------
print("📥 加载数据...")

# 分别读取 A站 和 B站 WPR 数据
df_wpr_a = pd.read_csv("a站点_风廓线雷达_清洗后数据.csv", parse_dates=['time'])
df_wpr_a["station"] = "A站"

df_wpr_b = pd.read_csv("b站点_风廓线雷达_清洗后数据.csv", parse_dates=['time'])
df_wpr_b["station"] = "B站"

# 合并两个站点
df_wpr = pd.concat([df_wpr_a, df_wpr_b], ignore_index=True)

# 加载模型 A 结果
df_a = pd.read_csv(MODEL_A_FILE, parse_dates=['time'])

# 期望列检查
for col in ['station','time','height_m','snr','wind_speed','vertical_velocity']:
    if col not in df_wpr.columns:
        raise ValueError(f"WPR 文件缺少必要列: {col}")

if 'TKE' not in df_a.columns:
    raise ValueError("模型 A 文件缺少列 'TKE'")

# 重命名模型 A 的 TKE 列为 TKE_A
df_a = df_a[['station','time','height_m','TKE']].rename(columns={'TKE':'TKE_A'})

# 合并 WPR 与模型 A（用于训练/拟合系数）
df = pd.merge(df_wpr, df_a, on=['station','time','height_m'], how='left')

# ----------------- 特征工程 -----------------
print("⚙️ 构造特征...")

df['f_inv_snr'] = safe_inv(df['snr'])
df['f_abs_w'] = np.abs(df['vertical_velocity'].astype(float))
df['f_wind'] = df['wind_speed'].astype(float)

def compute_shear(group):
    group = group.sort_values('height_m')
    u = group['wind_speed'].astype(float).values
    z = group['height_m'].astype(float).values
    if len(u) < 2: return pd.Series([np.nan]*len(group), index=group.index)
    if np.all(np.isnan(u)): return pd.Series([np.nan]*len(group), index=group.index)
    mask = np.isfinite(u)
    if mask.sum() < 2: return pd.Series([np.nan]*len(group), index=group.index)
    try:
        u_filled = np.interp(z, z[mask], u[mask])
    except Exception:
        u_filled = u
    du_dz = np.abs(np.gradient(u_filled, z, edge_order=2))
    return pd.Series(du_dz, index=group.index)

df['f_shear'] = df.groupby(['station','time']).apply(lambda g: compute_shear(g)).reset_index(level=[0,1], drop=True)
df = df.sort_values(['station','height_m','time'])
df['f_std_w_time'] = df.groupby(['station','height_m'])['vertical_velocity'].transform(lambda s: s.rolling(window=3, min_periods=1).std())

df['f_wind'].fillna(df['f_wind'].median(), inplace=True)
df['f_abs_w'].fillna(0.0, inplace=True)
df['f_inv_snr'].fillna(df['f_inv_snr'].median(), inplace=True)
df['f_shear'].fillna(0.0, inplace=True)
df['f_std_w_time'].fillna(0.0, inplace=True)

# ----------------- 构建设计矩阵 X -----------------
feature_cols = ['f_inv_snr','f_abs_w','f_wind','f_shear','f_std_w_time']
X = df[feature_cols].values
X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)

mask_train = np.isfinite(df['TKE_A'].values)
n_train = mask_train.sum()
if n_train < 10:
    print("⚠️ 可用于拟合的样本太少，退化为经验公式。")
    fit_coeffs = None
else:
    y = df.loc[mask_train, 'TKE_A'].values
    X_train = X[mask_train,:]
    print(f"🔧 使用 {n_train} 条样本拟合线性回归以拟合 TKE_A ...")
    coeffs, *_ = np.linalg.lstsq(X_train, y, rcond=None)
    fit_coeffs = coeffs
    print("拟合系数（intercept, inv_snr, |w|, wind, shear, std_w_time） =")
    print(np.round(fit_coeffs, 6))

# ----------------- 生成 TKE_B* -----------------
print("✨ 生成优化后的 TKE_B* ...")
if fit_coeffs is None:
    alpha, beta = 0.8, 0.5
    gamma = 0.05
    delta = 0.01
    snr = df['snr'].fillna(1e-3).astype(float)
    w = df['vertical_velocity'].fillna(0.0).astype(float)
    wind = df['wind_speed'].fillna(0.0).astype(float)
    df['TKE_B_star'] = (alpha * (snr.clip(lower=1e-3) ** (-beta)) +
                        gamma * np.abs(w) +
                        delta * (wind / 10.0))
else:
    df['TKE_B_star'] = X.dot(fit_coeffs)

df['TKE_B_star'] = np.clip(df['TKE_B_star'].astype(float), 0.0, None)

if fit_coeffs is not None and n_train > 0:
    train_rmse = rmse(df.loc[mask_train,'TKE_A'].values, df.loc[mask_train,'TKE_B_star'].values)
    train_mae  = mae(df.loc[mask_train,'TKE_A'].values, df.loc[mask_train,'TKE_B_star'].values)
    print(f"训练集拟合误差 -> RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")

# ----------------- 保存输出 -----------------
out_cols = ['station','time','height_m','snr','wind_speed','vertical_velocity','TKE_B_star']
out_df = df[out_cols].copy()
out_file = Path(OUTPUT_DIR)/"model_b_star_tke_estimation.csv"
out_df.to_csv(out_file, index=False)
print("✅ 已保存优化后模型 B* 结果：", out_file)

# ----------------- 可视化 -----------------
print("📊 绘制时高剖面图（按站点）...")
import matplotlib.dates as mdates

def plot_tke_profile_time_series(df_all, station, tke_col, ax, cmap='plasma', vmin=None, vmax=None):
    sub = df_all[df_all['station']==station].copy()
    if sub.empty: return
    pivot = sub.pivot_table(index='time', columns='height_m', values=tke_col, aggfunc='mean')
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    times = pivot.index
    heights = pivot.columns.values
    arr = pivot.T.values
    if vmin is None: vmin = np.nanpercentile(arr, 2)
    if vmax is None: vmax = np.nanpercentile(arr, 98)
    times_num = mdates.date2num(times.to_pydatetime())
    extent = [times_num[0], times_num[-1], heights.min(), heights.max()]
    im = ax.imshow(arr, aspect='auto', origin='lower', cmap=cmap, extent=extent,
                   interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_ylabel('高度 (m)')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    return im

stations = df['station'].unique()
n = len(stations)
fig, axes = plt.subplots(n, 2, figsize=(14, 4*n), sharex='col')
if n == 1: axes = np.array([axes])

combined_vals = []
for st in stations:
    a_vals = df_a[df_a['station']==st]['TKE_A'].values
    b_vals = df[df['station']==st]['TKE_B_star'].values
    combined_vals.extend(a_vals[np.isfinite(a_vals)])
    combined_vals.extend(b_vals[np.isfinite(b_vals)])
if len(combined_vals) == 0:
    vmin, vmax = 0.0, 1.0
else:
    vmin = np.nanpercentile(combined_vals, 5)
    vmax = np.nanpercentile(combined_vals, 95)

for i, st in enumerate(stations):
    axA = axes[i,0]; axB = axes[i,1]
    imA = plot_tke_profile_time_series(df, st, 'TKE_A' , axA, vmin=vmin, vmax=vmax)
    imB = plot_tke_profile_time_series(df, st, 'TKE_B_star', axB, vmin=vmin, vmax=vmax)
    axA.set_title(f"{st} - TKE (模型A 基准)")
    axB.set_title(f"{st} - TKE (模型B*)")
    fig.colorbar(imB, ax=[axA,axB], label='TKE', pad=0.02)

plt.xlabel('时间')
plt.tight_layout()
figfile = Path(OUTPUT_DIR)/"tke_time_height_comparison.png"
plt.savefig(figfile, dpi=200)
plt.close()
print("图像已保存：", figfile)

# ----------------- 诊断统计 -----------------
metrics = []
for st in stations:
    sub = df[df['station']==st]
    mask_valid = np.isfinite(sub['TKE_A']) & np.isfinite(sub['TKE_B_star'])
    metrics.append({
        'station': st,
        'n_samples': int(mask_valid.sum()),
        'rmse_A_vs_Bstar': rmse(sub.loc[mask_valid,'TKE_A'].values, sub.loc[mask_valid,'TKE_B_star'].values),
        'mae_A_vs_Bstar' : mae(sub.loc[mask_valid,'TKE_A'].values, sub.loc[mask_valid,'TKE_B_star'].values)
    })
met_df = pd.DataFrame(metrics)
met_df.to_csv(Path(OUTPUT_DIR)/"diagnostic_metrics_A_vs_Bstar.csv", index=False)
print("诊断指标已保存：", Path(OUTPUT_DIR)/"diagnostic_metrics_A_vs_Bstar.csv")
print(met_df)

print("全部完成 ✅")