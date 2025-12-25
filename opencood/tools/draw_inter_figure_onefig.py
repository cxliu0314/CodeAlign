import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# 设置科研风格的绘图参数
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2

# 从表格中提取的数据
models_data = {
    "HEAL": {"AP30": 91.89, "TP": 16.95, "color": "#1f77b4"},
    "CodeAlign One-to-one": {"AP30": 90.28, "TP": 2.70, "color": "#ff7f0e"},
    "CodeAlign Multi-Head": {"AP30": 89.63, "TP": 1.40, "color": "#2ca02c"},
    "CodeAlign Gated Modulation*": {"AP30": 89.85, "TP": 1.38, "color": "#d62728"}
}

# HEAL的TP-N关系数据
heal_N = [1, 2, 3, 4]
heal_TP = [0.23, 0.79, 2.56, 17.51]

# 其他方法的TP-N关系函数
tp_relations = {
    "CodeAlign One-to-one": lambda N: 0.3*N + 0.2*N*(N - 1),
    "CodeAlign Multi-Head": lambda N: 0.3*N + 0.2*N + 0.005 * (N - 1),
    "CodeAlign Gated Modulation": lambda N: 0.3*N + 0.2*N + 0.00006 * (N - 1)
}

# 创建图形和双横轴
fig, ax2 = plt.subplots(figsize=(10, 8))
ax1 = ax2.twiny()

# 设置科研配色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 第一步：绘制AP30-TP散点图
# 设置不同的标记形状和大小
marker_shapes = {
    "HEAL": "o",  # 圆形
    "CodeAlign One-to-one": "o",  # 圆形
    "CodeAlign Multi-Head": "s",  # 正方形（更大）
    "CodeAlign Gated Modulation*": "D"  # 星形
}

marker_sizes = {
    "HEAL": 300,
    "CodeAlign One-to-one": 300,
    "CodeAlign Multi-Head": 350,  # 更大的正方形
    "CodeAlign Gated Modulation*": 300  # 星形通常需要更大一些才明显
}

for i, (model, data) in enumerate(models_data.items()):
    marker = marker_shapes.get(model, "o")
    size = marker_sizes.get(model, 100)
    ax1.scatter(data["AP30"], data["TP"], color=data["color"], s=size, 
                marker=marker, label=model, zorder=5, edgecolors='black', linewidth=1)

# 第二步：绘制TP随group数量变化的虚线
N_range = np.linspace(1, 10, 50)

# 绘制HEAL的TP-N曲线
ax2.plot(heal_N, heal_TP, color=models_data["HEAL"]["color"], 
         linestyle='--', linewidth=2.5, label='HEAL TP-N', zorder=4)

# 绘制其他方法的TP-N曲线
for model_name, func in tp_relations.items():
    color = None
    # 匹配颜色
    for key in models_data:
        if model_name in key:
            color = models_data[key]["color"]
            break
    if color is None:
        color = colors[len(tp_relations) - list(tp_relations.keys()).index(model_name)]
    
    TP_values = [func(N) for N in N_range]
    # 为Gated Modulation设置更细的虚线
    if "Multi-Head" in model_name:
        ax2.plot(N_range, TP_values, color=color, linestyle='--', 
                 linewidth=3, label=f'{model_name} TP-N', zorder=3)
    else:
        ax2.plot(N_range, TP_values, color=color, linestyle='--', 
                 linewidth=2, label=f'{model_name} TP-N', zorder=3)

# 设置坐标轴标签和范围
ax1.set_xlabel('AP30 (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Training Parameters (M)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Group Number (N)', fontsize=14, fontweight='bold')

# 设置坐标轴范围
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 20)
ax2.set_xlim(1, 10)

# 设置刻度
ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))

# 添加网格
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 合并图例 - 都放在左上角
# 创建合并的图例元素
legend_elements = []

# 添加散点图的图例元素（使用正确的标记形状和大小）
for model, data in models_data.items():
    marker = marker_shapes.get(model, "o")
    legend_elements.append(
        Line2D([0], [0], marker=marker, color='w', 
               markerfacecolor=data["color"], markersize=8, 
               markeredgecolor='black', markeredgewidth=1,
               label=model)
    )

# 添加线条图例（TP-N曲线）
legend_elements.append(
    Line2D([0], [0], color=models_data["HEAL"]["color"], linestyle='--', 
           linewidth=2, label='HEAL TP-N')
)
legend_elements.append(
    Line2D([0], [0], color=models_data["CodeAlign One-to-one"]["color"], linestyle='--', 
           linewidth=2, label='One-to-one TP-N')
)
legend_elements.append(
    Line2D([0], [0], color=models_data["CodeAlign Multi-Head"]["color"], linestyle='--', 
           linewidth=2, label='Multi-Head TP-N')
)
# Gated Modulation图例使用更细的线宽
legend_elements.append(
    Line2D([0], [0], color=models_data["CodeAlign Gated Modulation*"]["color"], linestyle='--', 
           linewidth=2, label='Gated Modulation TP-N')
)

# 创建合并的图例
combined_legend = ax1.legend(handles=legend_elements, loc='upper left', 
                            frameon=True, fancybox=True, shadow=True, 
                            fontsize=10, ncol=1)

# 设置标题
plt.title('Model Performance vs Training Parameters and Group Number', 
          fontsize=16, fontweight='bold', pad=20)

# 调整布局
plt.tight_layout()

# 可选：保存为高质量图片
plt.savefig('inter_group_alignment_analysis.png', dpi=300, bbox_inches='tight')
