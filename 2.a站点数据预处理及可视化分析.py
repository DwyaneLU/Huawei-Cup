import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_rad_data(file_path):
    """
    解析风廓线雷达数据文件（WNDRAD/ROBS格式），返回DataFrame
    """
    data = []
    current_cycle = None
    timestamp_str = None

    # 从文件名提取时间戳
    file_name = os.path.basename(file_path)
    time_match = re.search(r'(\d{14})', file_name)
    if time_match:
        timestamp_str = time_match.group(1)
        try:
            time_obj = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except ValueError as e:
            print(f"文件 {file_name} 的时间戳解析失败: {e}")
            return pd.DataFrame()
    else:
        print(f"无法从文件名 {file_name} 中提取时间戳")
        return pd.DataFrame()

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.readlines()

        for line_num, line in enumerate(content, 1):
            line = line.strip()
            if not line:
                continue

            # 识别扫描周期
            if "RAD FIRST" in line or "ROBS FIRST" in line:
                current_cycle = "FIRST"
                continue
            elif "RAD SECOND" in line or "ROBS SECOND" in line:
                current_cycle = "SECOND"
                continue
            elif "RAD THIRD" in line or "ROBS THIRD" in line:
                current_cycle = "THIRD"
                continue
            elif "RAD FOURTH" in line or "ROBS FOURTH" in line:
                current_cycle = "FOURTH"
                continue
            elif "RAD FIFTH" in line or "ROBS FIFTH" in line:
                current_cycle = "FIFTH"
                continue
            elif "NNNN" in line:
                current_cycle = None
                continue

            # 只处理 FIRST 周期
            if current_cycle != "FIRST":
                continue

            # 匹配数据行：高度 SNR 风速 垂直速度
            pattern = r'^(\d{5})\s+([0-9./]{5,7})\s+([0-9./]{5,7})\s+([0-9./]{5,7})'
            match = re.match(pattern, line)
            if match:
                height = int(match.group(1))
                snr_str = match.group(2).strip()
                wind_speed_str = match.group(3).strip()
                vert_vel_str = match.group(4).strip()

                def parse_float(val):
                    if '/////' in val or val == '9999.0' or val == '':
                        return np.nan
                    try:
                        return float(val)
                    except:
                        return np.nan

                snr = parse_float(snr_str)
                wind_speed = parse_float(wind_speed_str)
                vertical_velocity = parse_float(vert_vel_str)

                data.append({
                    'time': time_obj,
                    'height_m': height,
                    'snr': snr,
                    'wind_speed': wind_speed,
                    'vertical_velocity': vertical_velocity
                })

    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)


# ================== 主程序 ==================

# 修改为您的实际文件夹路径
folder_path = 'a站点风廓线雷达数据'

# 检查文件夹是否存在
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"文件夹 {folder_path} 不存在，请检查路径。")

# ✅ 获取所有 .TXT 文件，并过滤出风廓线雷达数据（排除微波辐射计）
all_files = [
    f for f in os.listdir(folder_path)
    if f.upper().endswith('.TXT')
    and 'MWR' not in f.upper()
    and 'RADIOMETER' not in f.upper()
    and '辐射计' not in f
]

print(f"筛选出 {len(all_files)} 个风廓线雷达数据文件：")
for f in all_files:
    print(f"  - {f}")

if len(all_files) == 0:
    raise FileNotFoundError("未找到任何风廓线雷达的TXT文件，请检查文件名或排除微波辐射计文件。")

# 逐个解析文件
all_dfs = []
for file_name in all_files:
    file_path = os.path.join(folder_path, file_name)
    print(f"正在解析: {file_name}")
    df = parse_rad_data(file_path)
    if not df.empty:
        print(f"  -> 解析成功，包含 {len(df)} 行数据。")
        all_dfs.append(df)
    else:
        print(f"  -> 解析失败或无有效数据。")

# 合并所有数据
if not all_dfs:
    raise ValueError("所有文件均未解析出有效数据，请检查文件格式和内容。")

df_combined = pd.concat(all_dfs, ignore_index=True)

# ✅ 检查 'time' 列
if 'time' not in df_combined.columns:
    raise KeyError("解析后的数据中缺少 'time' 列，请检查文件名格式和解析逻辑。")

# 排序
df_combined.sort_values(['time', 'height_m'], inplace=True)
df_combined.reset_index(drop=True, inplace=True)

print(f"\n数据合并完成，总数据量: {len(df_combined)} 行")
print("数据预览:")
print(df_combined.head(10))

# ========== 数据清洗 ==========
# 去除风速和垂直速度均为 NaN 的行
df_cleaned = df_combined.dropna(subset=['wind_speed', 'vertical_velocity'], how='all')

# 异常值处理
df_cleaned.loc[df_cleaned['wind_speed'] > 100, 'wind_speed'] = np.nan
df_cleaned.loc[df_cleaned['vertical_velocity'].abs() > 10, 'vertical_velocity'] = np.nan

# ✅ 关键修复：处理 (time, height_m) 重复问题
# 对相同时间和高度的数据取平均值，确保唯一性
df_cleaned = df_cleaned.groupby(['time', 'height_m'], as_index=False).agg({
    'snr': 'mean',
    'wind_speed': 'mean',
    'vertical_velocity': 'mean'
})

print(f"去重聚合后数据量: {len(df_cleaned)} 行")

# ========== 可视化 ==========

# 1. 时间-高度剖面图 (Hovmöller Diagram)
try:
    # 使用 pivot_table 避免重复索引错误
    pivot_ws = df_cleaned.pivot_table(
        index='time',
        columns='height_m',
        values='wind_speed',
        aggfunc='mean'  # 冗余保护
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_ws.T, cmap='viridis', cbar_kws={'label': '风速 (m/s)'}, xticklabels=10, yticklabels=10)
    plt.title('A站 (58235) 风速垂直结构随时间演变')
    plt.xlabel('时间 (UTC)')
    plt.ylabel('高度 (m)')
    plt.gca().invert_yaxis()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"绘制Hovmöller图时出错: {e}")

# 2. 垂直廓线图 (Vertical Profile)
try:
    target_time = df_cleaned['time'].iloc[0]  # 可改为 df_cleaned['time'].max() 看最新时刻
    df_time = df_cleaned[df_cleaned['time'] == target_time]

    if df_time.empty:
        print("无法绘制垂直廓线：指定时间无数据。")
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # 绘制风速
        ax1.set_xlabel('风速 (m/s)', color='tab:blue')
        ax1.set_ylabel('高度 (m)', color='tab:blue')
        ax1.plot(df_time['wind_speed'], df_time['height_m'], color='tab:blue', marker='o', label='风速')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.tick_params(axis='x', labelcolor='tab:blue')

        # 第二个x轴：垂直速度
        ax2 = ax1.twiny()
        ax2.set_xlabel('垂直速度 (m/s)', color='tab:red')
        ax2.plot(df_time['vertical_velocity'], df_time['height_m'], color='tab:red', marker='s', label='垂直速度')
        ax2.tick_params(axis='x', labelcolor='tab:red')

        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title(f'A站 (58235) {target_time.strftime("%Y-%m-%d %H:%M")} 垂直廓线')
        fig.tight_layout()
        plt.show()

except Exception as e:
    print(f"绘制垂直廓线图时出错: {e}")

# ✅ 保存清洗后的数据
output_csv = 'a站点_风廓线雷达_清洗后数据.csv'
df_cleaned.to_csv(output_csv, index=False, encoding='utf-8-sig')  # utf-8-sig 避免Excel乱码
print(f"数据预处理与可视化完成，清洗后数据已保存至: {output_csv}")