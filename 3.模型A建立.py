import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
from scipy.interpolate import interp1d
from pathlib import Path

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. 解析 MWR 数据 ==================
def parse_mwr_data(file_path, station_name):
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        if any(kw in line for kw in ['Record', 'DateTime', 'N,', '----', '时间', '1120', 'time']):
            continue

        parts = re.split(r'\s+', line)
        if len(parts) < 11:
            continue

        try:
            record_id = parts[0]
            date_str = parts[1]
            time_str = parts[2]
            dt_str = date_str + ' ' + time_str
            time_obj = datetime.strptime(dt_str, "%Y/%m/%d %H:%M")

            sur_tem = float(parts[3])
            sur_hum = float(parts[4])
            sur_pre = float(parts[5])
            tir = float(parts[6])

            # ⚠️ 关键修改：rain 强制 float → int，异常赋 NaN
            try:
                rain = int(float(parts[7]))
            except:
                rain = np.nan

            cloud_base = float(parts[8]) if parts[8] not in ['0', ''] else np.nan
            vint = float(parts[9])
            lqint = float(parts[10])

            data.append({
                'time': time_obj,
                'station': station_name,
                'SurTem': sur_tem,
                'SurHum': sur_hum,
                'SurPre': sur_pre,
                'Tir': tir,
                'Rain': rain,
                'CloudBase': cloud_base,
                'Vint(mm)': vint,
                'Lqint(mm)': lqint
            })
        except Exception as e:
            print(f"跳过第 {line_num + 1} 行: {line[:40]}... | 错误: {e}")
            continue
    return pd.DataFrame(data)


# ================== 2. 加载 WPR 数据 ==================
def load_wpr_data(file_path, station_name):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df['station'] = station_name
    return df


# ================== 3. 时间对齐 + 插值融合 ==================
def align_and_fuse(wpr_df, mwr_df, method='nearest', tolerance='5min'):
    """
    将 MWR 数据按时间对齐到 WPR 数据点，并进行插值
    """
    # 确保时间列是 datetime 类型
    wpr_df['time'] = pd.to_datetime(wpr_df['time'])
    mwr_df['time'] = pd.to_datetime(mwr_df['time'])

    # 只保留数值列，避免 'station' 等字符串列影响 groupby
    numeric_cols = ['Vint(mm)', 'Lqint(mm)', 'SurTem', 'SurHum']
    mwr_numeric = mwr_df[['time'] + numeric_cols].copy()

    # 去重：如果同一时间有多条，取均值
    mwr_numeric = mwr_numeric.groupby('time')[numeric_cols].mean().reset_index()

    # 设置时间索引
    mwr_interp = mwr_numeric.set_index('time')

    # 使用 reindex 对齐
    mwr_aligned = mwr_interp.reindex(
        wpr_df['time'],
        method=method,
        tolerance=pd.Timedelta(tolerance)
    )

    # 合并数据
    fused = wpr_df.copy()
    for col in numeric_cols:
        fused[col] = mwr_aligned[col].values

    return fused


# ================== 4. 构建模型a：TKE 估算 ==================
def compute_tke_model_a(df):
    """
    模型a: 基于 SNR、垂直速度 w、MWR 环境因子估算 TKE
    公式: TKE = α * SNR^(-β) + γ * |w| + δ * (Lqint + Vint/10)
    参数通过经验设定（可后续用机器学习优化）
    """
    alpha, beta = 0.8, 0.5
    gamma = 0.05
    delta = 0.01  # LWP 影响较小，但高液态水抑制湍流

    snr = df['snr'].fillna(1e-3)
    w = df['vertical_velocity'].fillna(0.0)
    lqint = df['Lqint(mm)'].fillna(0.0)
    vint = df['Vint(mm)'].fillna(0.0)

    # 避免除零
    snr = np.where(snr <= 0, 1e-3, snr)

    # 计算 TKE
    tke = (alpha * (snr ** (-beta)) +
           gamma * np.abs(w) +
           delta * (lqint + vint / 10.0))

    # 添加稳定性修正：高湿度/高LWP → 抑制湍流
    stability_factor = 1.0 / (1.0 + 0.01 * lqint + 0.005 * (100 - df['SurHum'].fillna(50)))
    tke = tke * stability_factor

    return np.clip(tke, 0, None)  # TKE >= 0


# ================== 主程序 ==================
if __name__ == "__main__":
    print("🚀 开始构建模型a：WPR+MWR 融合估算 TKE")

    # --- 加载数据 ---
    print("📁 加载 MWR 数据...")
    df_mwr_a = parse_mwr_data('a站点微波辐射计数据.txt', 'A站')
    df_mwr_b = parse_mwr_data('b站点微波辐射计数据.txt', 'B站')

    print("📁 加载 WPR 数据...")
    df_wpr_a = load_wpr_data('a站点_风廓线雷达_清洗后数据.csv', 'A站')
    df_wpr_b = load_wpr_data('b站点_风廓线雷达_清洗后数据.csv', 'B站')

    # --- 数据融合 ---
    print("🔗 融合 A站 WPR+MWR 数据...")
    df_fused_a = align_and_fuse(df_wpr_a, df_mwr_a)
    df_fused_a['TKE'] = compute_tke_model_a(df_fused_a)

    print("🔗 融合 B站 WPR+MWR 数据...")
    df_fused_b = align_and_fuse(df_wpr_b, df_mwr_b)
    df_fused_b['TKE'] = compute_tke_model_a(df_fused_b)

    # 合并所有数据
    df_all = pd.concat([df_fused_a, df_fused_b], ignore_index=True)
    df_all.sort_values(['station', 'time', 'height_m'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    print(f"✅ 模型a完成！共处理 {len(df_all)} 条融合数据。")

    # --- 可视化 TKE 垂直结构 ---
    stations = df_all['station'].unique()
    fig, axes = plt.subplots(len(stations), 1, figsize=(14, 6), sharex=True)
    if len(stations) == 1:
        axes = [axes]

    for idx, station in enumerate(stations):
        ax = axes[idx]
        df_sub = df_all[df_all['station'] == station]
        df_pivot = df_sub.pivot_table(index='time', columns='height_m', values='TKE', aggfunc='mean')

        im = ax.imshow(df_pivot.T, aspect='auto', cmap='plasma', origin='lower',
                      extent=[df_pivot.index[0], df_pivot.index[-1],
                              df_pivot.columns.min(), df_pivot.columns.max()],
                      interpolation='none')
        ax.set_title(f'{station} 湍流强度 TKE 垂直结构 (模型a)')
        ax.set_ylabel('高度 (m)')
        fig.colorbar(im, ax=ax, label='TKE ')

    plt.xlabel('时间')
    plt.tight_layout()
    plt.show()

    # --- 保存结果 ---
    output_file = 'model_a_tke_estimation.csv'
    df_all.to_csv(output_file, index=False)
    print(f"💾 结果已保存至: {output_file}")

    # --- 输出统计摘要 ---
    print("\n📊 模型a TKE 统计摘要:")
    print(df_all.groupby('station')['TKE'].agg(['mean', 'std', 'min', 'max']))