import matplotlib.pyplot as plt
import matplotlib.patches as patches


plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 正常显示负号

# 创建画布
fig, ax = plt.subplots(figsize=(6,8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# 三个层级的信息
layers = [
    {"y": 0.5, "height": 3.0, "label": "基础层\n(观测支撑)",
     "items": "风速、温度、湿度、谱宽\n(风廓线雷达、微波辐射计)"},
    {"y": 3.6, "height": 3.0, "label": "诊断层\n(机理诊断)",
     "items": "风切变 S、Ri、TKE、ε、TI"},
    {"y": 6.7, "height": 3.0, "label": "应用层\n(航路规划)",
     "items": "航路风险等级、调整率、延误/改航比例\n预警触发次数、PIREPs 校验"}
]

# 绘制矩形框并加文字
for layer in layers:
    rect = patches.FancyBboxPatch((1, layer["y"]), 8, layer["height"],
                                  boxstyle="round,pad=0.3", linewidth=1.2, edgecolor='black', facecolor='#e0f7fa')
    ax.add_patch(rect)
    # 居中文字
    ax.text(5, layer["y"] + layer["height"]*0.62, layer["label"],
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, layer["y"] + layer["height"]*0.28, layer["items"],
            ha='center', va='center', fontsize=10)

# 添加箭头
ax.annotate('', xy=(5, 9.9), xytext=(5, 9.4), arrowprops=dict(arrowstyle='->', linewidth=1.2))
ax.annotate('', xy=(5, 6.4), xytext=(5, 5.9), arrowprops=dict(arrowstyle='->', linewidth=1.2))

# 标题与说明
ax.text(5, 11.3, "三层指标体系结构图：基础层 → 诊断层 → 应用层", ha='center', va='center', fontsize=14, fontweight='semibold')


# 保存和展示
plt.savefig("three_layer_pyramid.png", dpi=200, bbox_inches='tight')
plt.show()