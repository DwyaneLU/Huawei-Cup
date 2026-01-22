import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. 加载 WPR 数据 ==================
def load_wpr_data(file_path, station_name):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df['station'] = station_name
    return df

# ================== 2. 构建模型b：TKE 估算 ==================
def compute_tke_model_b(df):
    """
    模型b: 仅基于 WPR 数据进行 TKE 估算
    公式: TKE_b = α * SNR^(-β) + γ * |w| + δ * (0.5 * wind_speed^2)
    其中:
        - SNR (信噪比) 表征观测信号质量，与湍流复杂性相关
        - 垂直速度 w 直接反映垂直扰动强度
        - 风速的平方项代表动能贡献
    """
    alpha, beta = 0.6, 0.4
    gamma = 0.06
    delta = 0.005

    snr = df['snr'].fillna(1e-3)
    w = df['vertical_velocity'].fillna(0.0)
    wind = df['wind_speed'].fillna(0.0)

    # 避免除零
    snr = np.where(snr <= 0, 1e-3, snr)

    tke = (alpha * (snr ** (-beta)) +
           gamma * np.abs(w) +
           delta * (0.5 * wind ** 2))

    return np.clip(tke, 0, None)  # TKE >= 0

# ================== 主程序 ==================
if __name__ == "__main__":
    print("🚀 开始构建模型b：仅WPR时空建模估算 TKE")

    # --- 加载数据 ---
    print("📁 加载 WPR 数据...")
    df_wpr_a = load_wpr_data('a站点_风廓线雷达_清洗后数据.csv', 'A站')
    df_wpr_b = load_wpr_data('b站点_风廓线雷达_清洗后数据.csv', 'B站')

    # --- 计算 TKE ---
    print("⚡ 计算 A站 TKE_b...")
    df_wpr_a['TKE_b'] = compute_tke_model_b(df_wpr_a)

    print("⚡ 计算 B站 TKE_b...")
    df_wpr_b['TKE_b'] = compute_tke_model_b(df_wpr_b)

    # --- 合并 ---
    df_all = pd.concat([df_wpr_a, df_wpr_b], ignore_index=True)
    df_all.sort_values(['station', 'time', 'height_m'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    print(f"✅ 模型b完成！共处理 {len(df_all)} 条数据。")

    # --- 可视化 TKE 垂直结构 ---
    stations = df_all['station'].unique()
    fig, axes = plt.subplots(len(stations), 1, figsize=(14, 6), sharex=True)
    if len(stations) == 1:
        axes = [axes]

    for idx, station in enumerate(stations):
        ax = axes[idx]
        df_sub = df_all[df_all['station'] == station]
        df_pivot = df_sub.pivot_table(index='time', columns='height_m', values='TKE_b', aggfunc='mean')

        im = ax.imshow(df_pivot.T, aspect='auto', cmap='plasma', origin='lower',
                      extent=[df_pivot.index[0], df_pivot.index[-1],
                              df_pivot.columns.min(), df_pivot.columns.max()],
                      interpolation='none')
        ax.set_title(f'{station} 湍流强度 TKE 垂直结构 (模型b)')
        ax.set_ylabel('高度 (m)')
        fig.colorbar(im, ax=ax, label='TKE ')

    plt.xlabel('时间')
    plt.tight_layout()
    plt.show()

    # --- 保存结果 ---
    output_file = 'model_b_tke_estimation.csv'
    df_all.to_csv(output_file, index=False)
    print(f"💾 结果已保存至: {output_file}")

    # --- 输出统计摘要 ---
    print("\n📊 模型b TKE 统计摘要:")
    print(df_all.groupby('station')['TKE_b'].agg(['mean', 'std', 'min', 'max']))