import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 参数（修改为你实际文件名） ----------
MODEL_A_FILE = "model_a_tke_estimation.csv"  # 或你的文件名
MODEL_B_FILE = "model_b_tke_estimation.csv"  # 或你的文件名（如果尚未生成，先运行模型B）
OUTPUT_DIR = "validation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 辅助指标函数 ----------
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

def bias(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    return np.mean(a[mask] - b[mask])

def corrcoef(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0,1]

# ---------- 加载数据 ----------
def load_model_csv(path, model_name):
    df = pd.read_csv(path, parse_dates=['time'])
    # 规范列名（如果列名不同，按需求改）
    # 需要： station, time, height_m, TKE
    # 如果没有 TKE（比如模型B还没写入 TKE），你可以把模型B输出列名调整后再运行
    if 'TKE' not in df.columns:
        raise ValueError(f"{path} 中找不到列 'TKE'，请先确保模型输出包含 TKE 列。")
    df = df[['station','time','height_m','TKE']].copy()
    df.rename(columns={'TKE': f'TKE_{model_name}'}, inplace=True)
    return df

df_a = load_model_csv(MODEL_A_FILE, "A")
df_b = load_model_csv(MODEL_B_FILE, "B")

# ---------- 合并 A,B（按 station,time,height） ----------
df = pd.merge(df_a, df_b, on=['station','time','height_m'], how='outer')
# 保留排序
df.sort_values(['station','time','height_m'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------- 直接互比（A vs B） ----------
metrics = []
for station in df['station'].unique():
    sub = df[df['station']==station]
    rm = rmse(sub['TKE_A'].values, sub['TKE_B'].values)
    ma = mae(sub['TKE_A'].values, sub['TKE_B'].values)
    bi = bias(sub['TKE_A'].values, sub['TKE_B'].values)
    co = corrcoef(sub['TKE_A'].values, sub['TKE_B'].values)
    metrics.append({'station':station, 'comparison':'A_vs_B', 'rmse':rm, 'mae':ma, 'bias':bi, 'corr':co})

# ---------- 用集合统计作为伪真值（ensemble） ----------
# ensemble mean & median
df['TKE_ensemble_mean'] = df[['TKE_A','TKE_B']].mean(axis=1, skipna=True)
df['TKE_ensemble_median'] = df[['TKE_A','TKE_B']].median(axis=1, skipna=True)

for station in df['station'].unique():
    sub = df[df['station']==station]
    # A vs ensemble_mean
    metrics.append({
        'station':station, 'comparison':'A_vs_ensemble_mean',
        'rmse': rmse(sub['TKE_A'].values, sub['TKE_ensemble_mean'].values),
        'mae': mae(sub['TKE_A'].values, sub['TKE_ensemble_mean'].values),
        'bias': bias(sub['TKE_A'].values, sub['TKE_ensemble_mean'].values),
        'corr': corrcoef(sub['TKE_A'].values, sub['TKE_ensemble_mean'].values)
    })
    # B vs ensemble_mean
    metrics.append({
        'station':station, 'comparison':'B_vs_ensemble_mean',
        'rmse': rmse(sub['TKE_B'].values, sub['TKE_ensemble_mean'].values),
        'mae': mae(sub['TKE_B'].values, sub['TKE_ensemble_mean'].values),
        'bias': bias(sub['TKE_B'].values, sub['TKE_ensemble_mean'].values),
        'corr': corrcoef(sub['TKE_B'].values, sub['TKE_ensemble_mean'].values)
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(Path(OUTPUT_DIR)/"metrics_summary.csv", index=False)
print("指标汇总已保存：", Path(OUTPUT_DIR)/"metrics_summary.csv")
print(metrics_df)

# ---------- 绘图诊断：时间序列 & 散点图（按站点抽样绘图） ----------
import matplotlib.dates as mdates

def plot_time_series_for_station(station, save_dir):
    sub = df[df['station']==station]
    # 取若干典型高度：选最常见的前6个高度
    top_heights = sub['height_m'].value_counts().index[:6].tolist()
    fig, axs = plt.subplots(len(top_heights), 1, figsize=(12, 2.5*len(top_heights)), sharex=True)
    for i,h in enumerate(top_heights):
        ssub = sub[sub['height_m']==h]
        axs[i].plot(ssub['time'], ssub['TKE_A'], label='TKE_A', alpha=0.8)
        axs[i].plot(ssub['time'], ssub['TKE_B'], label='TKE_B', alpha=0.8)
        axs[i].plot(ssub['time'], ssub['TKE_ensemble_mean'], '--', label='ensemble_mean', alpha=0.8)
        axs[i].set_ylabel(f'height={h}m')
        axs[i].legend()
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.suptitle(f'{station} - TKE 时间序列（若干高度）')
    plt.tight_layout()
    outfile = Path(save_dir)/f"{station}_time_series_sample.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("保存：", outfile)

def plot_scatter_A_vs_B(station, save_dir):
    sub = df[df['station']==station]
    plt.figure(figsize=(6,6))
    plt.scatter(sub['TKE_A'], sub['TKE_B'], s=10, alpha=0.6)
    mn = np.nanmin([sub['TKE_A'].min(), sub['TKE_B'].min()])
    mx = np.nanmax([sub['TKE_A'].max(), sub['TKE_B'].max()])
    plt.plot([mn,mx],[mn,mx],'k--',linewidth=1)
    plt.xlabel('TKE_A'); plt.ylabel('TKE_B')
    plt.title(f'{station} - TKE_A vs TKE_B 散点图')
    outfile = Path(save_dir)/f"{station}_scatter_A_vs_B.png"
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("保存：", outfile)

for station in df['station'].unique():
    plot_time_series_for_station(station, OUTPUT_DIR)
    plot_scatter_A_vs_B(station, OUTPUT_DIR)

# ---------- 垂直剖面差异图（按时间取若干时刻） ----------
def plot_vertical_profiles_diff(station, times_to_plot, save_dir):
    sub = df[df['station']==station]
    for t in times_to_plot:
        ssub = sub[sub['time']==t]
        if ssub.empty:
            continue
        ssub = ssub.sort_values('height_m')
        plt.figure(figsize=(6,5))
        plt.plot(ssub['TKE_A'], ssub['height_m'], '-o', label='A')
        plt.plot(ssub['TKE_B'], ssub['height_m'], '-o', label='B')
        plt.gca().invert_yaxis()
        plt.xlabel('TKE'); plt.ylabel('高度 (m)')
        plt.title(f'{station} Vertical profile at {t}')
        plt.legend()
        outfile = Path(save_dir)/f"{station}_profile_{t.strftime('%H%M%S')}.png"
        plt.savefig(outfile, dpi=200)
        plt.close()
        print("保存：", outfile)

# 选择每站最常见的 3 个时间点
for station in df['station'].unique():
    times = df[df['station']==station]['time'].value_counts().index[:3].tolist()
    plot_vertical_profiles_diff(station, times, OUTPUT_DIR)

# ---------- 留一高度 (leave-one-height-out) 验证（伪真值由其它高度插值或 ensemble 推断） ----------
# 思路：对于每个站点、每个时间，若某高度同时存在 A/B，则把该高度“隐藏”，用其它高度的 ensemble_mean
#      通过高度插值预测该高度的伪真值（注意：这仅检验垂直一致性，不是绝对真值）。
from scipy.interpolate import interp1d

loo_results = []
for station in df['station'].unique():
    ssub = df[df['station']==station]
    times = ssub['time'].unique()
    for t in times:
        tsub = ssub[ssub['time']==t].dropna(subset=['TKE_ensemble_mean'])
        if len(tsub) < 3:
            continue  # 插值需要至少 3 个高度点
        heights = tsub['height_m'].values
        ens_vals = tsub['TKE_ensemble_mean'].values
        # 构造插值函数（按高度），用 linear 并允许外推（fill_value="extrapolate")
        f_interp = interp1d(heights, ens_vals, kind='linear', fill_value="extrapolate", bounds_error=False)
        # 对每个被测高度做验证
        for idx, row in tsub.iterrows():
            h = row['height_m']
            pseudo_truth = f_interp(h)  # 用其它高度插值预测（注意：这里包含自身——为严谨可使用 leave-one-point interp；简单起见用整体插值）
            # 若想严格 leave-one: 可用 heights_except = heights[heights!=h] 等进行再插值（我在下面实现严格 leave-one）
        # 严格 leave-one 实现：
        for i in range(len(heights)):
            h_i = heights[i]
            mask = np.ones(len(heights), dtype=bool)
            mask[i] = False
            if mask.sum() < 2:
                continue
            try:
                f_lo = interp1d(heights[mask], ens_vals[mask], kind='linear', fill_value="extrapolate", bounds_error=False)
                pseudo = f_lo(h_i)
            except Exception:
                continue
            # 查找行对应 A,B 值
            row = tsub[tsub['height_m']==h_i].iloc[0]
            val_a = row['TKE_A']
            val_b = row['TKE_B']
            loo_results.append({
                'station':station, 'time':t, 'height_m':h_i,
                'pseudo_truth': float(pseudo), 'A': float(val_a) if pd.notna(val_a) else np.nan,
                'B': float(val_b) if pd.notna(val_b) else np.nan
            })

loo_df = pd.DataFrame(loo_results)
# 指标
loo_metrics = []
for station in loo_df['station'].unique():
    s = loo_df[loo_df['station']==station]
    loo_metrics.append({
        'station':station, 'method':'leave_one_height_pseudo',
        'A_rmse': rmse(s['A'].values, s['pseudo_truth'].values),
        'A_mae': mae(s['A'].values, s['pseudo_truth'].values),
        'B_rmse': rmse(s['B'].values, s['pseudo_truth'].values),
        'B_mae': mae(s['B'].values, s['pseudo_truth'].values),
        'n_samples': len(s)
    })
loo_metrics_df = pd.DataFrame(loo_metrics)
loo_metrics_df.to_csv(Path(OUTPUT_DIR)/"loo_metrics.csv", index=False)
loo_df.to_csv(Path(OUTPUT_DIR)/"loo_all_samples.csv", index=False)
print("留一高度结果已保存：", Path(OUTPUT_DIR)/"loo_metrics.csv", Path(OUTPUT_DIR)/"loo_all_samples.csv")
print(loo_metrics_df)

# ---------- 保存合并数据以便后续分析 ----------
df.to_csv(Path(OUTPUT_DIR)/"merged_modelA_modelB.csv", index=False)
print("合并数据已保存：", Path(OUTPUT_DIR)/"merged_modelA_modelB.csv")

print("所有分析完成。请检查目录：", OUTPUT_DIR)