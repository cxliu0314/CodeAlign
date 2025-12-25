import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 高级科研图表设置
mpl.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# 从参考代码中提取的数据
models_data = {
    "HEAL": {"AP30": 91.89, "TP": 16.95, "color": "#1f77b4"},
    "CodeAlign \n[One-to-one]": {"AP30": 90.28, "TP": 2.70, "color": "#ff7f0e"},
    "CodeAlign \n[Multi-Head]": {"AP30": 89.63, "TP": 1.40, "color": "#2ca02c"},
    "CodeAlign \n[Channel Modulation]": {"AP30": 89.85, "TP": 1.38, "color": "#d62728"}
}

# HEAL的TP-N关系数据
heal_N = [1, 2, 3, 4, 5]
heal_TP = [0, 0.23, 0.79, 2.56, 17.51]

# 其他方法的TP-N关系函数
tp_relations = {
    "CodeAlign \n[One-to-one]": lambda N: 0.3*N + 0.2*N*(N - 1),
    "CodeAlign \n[Multi-Head]": lambda N: 0.3*N + 0.2*N + 0.005 * (N - 1),
    "CodeAlign \n[Channel Modulation]": lambda N: 0.3*N + 0.2*N + 0.00006 * (N - 1)
}

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 设置不同的标记形状和大小
marker_shapes = {
    "HEAL": "o",  # 圆形
    "CodeAlign \n[One-to-one]": "^",  # 圆形
    "CodeAlign \n[Multi-Head]": "s",  # 正方形
    "CodeAlign \n[Channel Modulation]": "D"  # 菱形
}

marker_sizes = {
    "HEAL": 300,
    "CodeAlign \n[One-to-one]": 300,
    "CodeAlign \n[Multi-Head]": 350,  # 更大的正方形
    "CodeAlign \n[Channel Modulation]": 300  # 菱形
}

# 左图：AP30 vs TP散点图
for model, data in models_data.items():
    marker = marker_shapes.get(model, "o")
    size = marker_sizes.get(model, 300)
    ax1.scatter(data["TP"], data["AP30"], color=data["color"], s=size, 
                marker=marker, label=model, zorder=5, edgecolors='black', linewidth=1.2)

ax1.set_ylabel('AP30 (%)', fontweight='bold')
ax1.set_xlabel('Training Parameters (M)', fontweight='bold')
# ax1.set_title('(a) AP30 vs Training Parameters', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right', markerscale=0.6, fontsize=10)  # 设置图例标记大小为原图的1/3，约100
ax1.set_ylim(0, 110)
ax1.set_xlim(0, 20)

# 右图：TP随Group数量变化（使用参考代码的公式）
N_range = np.linspace(1, 10, 50)  # Group数量从1到10

# 绘制HEAL的TP-N曲线（使用离散点数据）
ax2.plot(heal_N, heal_TP, color=models_data["HEAL"]["color"], 
         linestyle='--', linewidth=2.5, label='HEAL', zorder=4, marker='o', markersize=6)

# 绘制其他方法的TP-N曲线（使用lambda函数）
for model_name, func in tp_relations.items():
    color = None
    # 匹配颜色
    for key in models_data:
        if model_name in key:
            color = models_data[key]["color"]
            break
    if color is None:
        # 如果找不到匹配，使用默认颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        color = colors[len(tp_relations) - list(tp_relations.keys()).index(model_name)]
    
    TP_values = [func(N) for N in N_range]
    
    # 为不同方法设置不同的线型
    if "Channel Modulation" in model_name:
        ax2.plot(N_range, TP_values, color=color, linestyle='--', 
                 linewidth=1.5, label=model_name, zorder=3)
    elif "Multi-Head" in model_name:
        ax2.plot(N_range, TP_values, color=color, linestyle='--', 
                 linewidth=2.5, label=model_name, zorder=3)
    else:
        ax2.plot(N_range, TP_values, color=color, linestyle='--', 
                 linewidth=2, label=model_name, zorder=3)

ax2.set_xlabel('Number of Groups', fontweight='bold')
ax2.set_ylabel('Training Parameters (M)', fontweight='bold')
# ax2.set_title('(b) Training Parameters Scaling with Group Count', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=9)
ax2.set_xlim(1, 10)
ax2.set_ylim(0, 20)

# 在图表下方添加小标题
plt.figtext(0.27, -0.01, '(a) AP30 vs Training Parameters', ha='center', fontweight='bold', fontsize=16)
plt.figtext(0.77, -0.01, '(b) Training Parameters Scaling with Group Number', ha='center', fontweight='bold', fontsize=16)

# 整体标题
plt.suptitle('Model Ablation Analysis for Inter-group Alignment', 
             fontsize=20, fontweight='bold', y=0.93)

plt.tight_layout()
plt.savefig('inter_group_analysis_comprehensive.png', dpi=300, bbox_inches='tight')