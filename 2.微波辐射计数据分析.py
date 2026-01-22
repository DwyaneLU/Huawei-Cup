import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import re

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def parse_mwr_data(file_path, station_name):
    """
    解析微波辐射计 .txt 文件
    """
    data = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # 跳过表头或无效行
            if any(kw in line for kw in ['Record', 'DateTime', 'N,', '----', '时间']):
                continue

            parts = re.split(r'\s+', line)
            if len(parts) < 11:
                continue  # 至少需要前11列

            try:
                # 提取字段（根据实际顺序）
                record_id = parts[0]
                date_str = parts[1]  # "2025/8/2"
                time_str = parts[2]  # "0:00"
                dt_str = date_str + ' ' + time_str
                time_obj = datetime.strptime(dt_str, "%Y/%m/%d %H:%M")

                sur_tem = float(parts[3])   # 地面温度
                sur_hum = float(parts[4])   # 地面湿度
                sur_pre = float(parts[5])   # 气压
                tir = float(parts[6])       # 亮温 (float)
                rain = int(float(parts[7])) # 降水标志，可能是 "0.0"/"1.0"
                cloud_base = float(parts[8]) if parts[8] not in ['0', ''] else np.nan
                vint = float(parts[9])      # 水汽总量
                lqint = float(parts[10])    # 液态水路径

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
                print(f"跳过第 {line_num + 1} 行: {line[:60]}... | 错误: {e}")
                continue
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)


# ================== 主程序 ==================
file_a = r'a站点微波辐射计数据.txt'  # ✅ 修改为你的实际路径
file_b = r'b站点微波辐射计数据.txt'

print("正在解析 A 站数据...")
df_a = parse_mwr_data(file_a, station_name='A站')

print("正在解析 B 站数据...")
df_b = parse_mwr_data(file_b, station_name='B站')

# 合并
if df_a.empty and df_b.empty:
    raise ValueError("未解析到任何有效数据，请检查文件格式。")

df_combined = pd.concat([df_a, df_b], ignore_index=True)
df_combined.sort_values(['station', 'time'], inplace=True)
df_combined.reset_index(drop=True, inplace=True)

print(f"✅ 解析成功：A站 {len(df_a)} 条，B站 {len(df_b)} 条。")

# ========== 可视化 ==========
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for station in ['A站', 'B站']:
    df_sub = df_combined[df_combined['station'] == station]
    color = 'tab:blue' if station == 'A站' else 'tab:orange'
    alpha = 0.8

    axes[0].plot(df_sub['time'], df_sub['Lqint(mm)'], label=f'{station} LWP', color=color, alpha=alpha)
    axes[1].plot(df_sub['time'], df_sub['Vint(mm)'], label=f'{station} Vint', color=color, alpha=alpha)

axes[0].set_ylabel('液态水路径 LWP (mm)')
axes[1].set_ylabel('水汽总量 Vint (mm)')
axes[0].legend(), axes[1].legend()
axes[0].grid(True, alpha=0.3), axes[1].grid(True, alpha=0.3)
axes[0].set_title('LWP'), axes[1].set_title('Vint')
axes[1].set_xlabel('时间')

plt.suptitle('A站 vs B站 微波辐射计 LWP & Vint 对比')
plt.tight_layout()
plt.show()

# 地面温湿对比
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for station in ['A站', 'B站']:
    df_sub = df_combined[df_combined['station'] == station]
    color = 'tab:blue' if station == 'A站' else 'tab:orange'
    ax1.plot(df_sub['time'], df_sub['SurTem'], label=station, color=color)
    ax2.plot(df_sub['time'], df_sub['SurHum'], label=station, color=color)

ax1.set_ylabel('温度 (°C)')
ax2.set_ylabel('湿度 (%)')
ax1.legend(), ax2.legend()
ax1.grid(True, alpha=0.3), ax2.grid(True, alpha=0.3)
ax1.set_title('地面温度'), ax2.set_title('地面湿度')
ax2.set_xlabel('时间')
plt.suptitle('A站与B站 地面气象要素对比')
plt.tight_layout()
plt.show()

print("📊 数据解析与可视化完成！")