import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 全局样式配置（字体加大） ----------------------
plt.rcParams['figure.facecolor'] = 'white'   # 画布背景色（白色）
plt.rcParams['axes.facecolor'] = '#DCDCDC'   # 子图背景色（浅灰色）
plt.rcParams['grid.color'] = 'white'         # 网格色（白色，与子图背景对比）
plt.rcParams['grid.linewidth'] = 0.8         # 网格线粗细
plt.rcParams['lines.linewidth'] = 2.0        # 曲线基础粗细
plt.rcParams['font.size'] = 22               # 全局字体
plt.rcParams['axes.labelpad'] = 6            # 坐标轴标签间距

# ---------------------- 2. 数据 ----------------------
pose_error_a = np.linspace(0, 0.6, 4)
ap30_late = [0.8447, 0.8378, 0.7725, 0.7018]
ap30_heal = [0.8245, 0.8252, 0.8199, 0.8103]
ap30_codealign = [0.8719, 0.8702, 0.8586, 0.843]
ap30_nocolla = [0.7996, 0.7996, 0.7996, 0.7996]

pose_error_b = np.linspace(0, 0.6, 4)
ap50_late = [0.8012, 0.7415, 0.6138, 0.5629]
ap50_heal = [0.8103, 0.808, 0.7959, 0.7836]
ap50_codealign = [0.853, 0.8464, 0.8197, 0.7952]
ap50_nocolla = [0.7797, 0.7797, 0.7797, 0.7797]

# ---------------------- 3. 创建画布与子图（高度减小） ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ---------------------- 4. 子图(a) - AP30 ----------------------
ax1.plot(pose_error_a, ap30_codealign, color='#E74C3C',
         markersize=6, linewidth=3, label='CodeAlign')
ax1.plot(pose_error_a, ap30_heal, color='#3498DB',
         markersize=6, linewidth=3, label='HEAL')
ax1.plot(pose_error_a, ap30_late, color='#F39C12',
         markersize=6, linewidth=3, label='Late Fusion')
ax1.plot(pose_error_a, ap30_nocolla, color='#4A4A4A',
         marker='None', linestyle='--', linewidth=3, label='No Collaboration')

ax1.set_xlabel('Pose Error (Std)')
ax1.set_ylabel('AP30')
ax1.grid(True, linestyle='-', alpha=0.8)
ax1.set_ylim(0.7, 0.88)
ax1.set_xticks([0.0, 0.2, 0.4, 0.6])
ax1.set_xlim(-0.05, 0.65)

# ---------------------- 5. 子图(b) - AP50 ----------------------
ax2.plot(pose_error_b, ap50_codealign, color='#E74C3C',
         markersize=6, linewidth=3, label='CodeAlign')
ax2.plot(pose_error_b, ap50_heal, color='#3498DB',
         markersize=6, linewidth=3, label='HEAL')
ax2.plot(pose_error_b, ap50_late, color='#F39C12',
         markersize=6, linewidth=3, label='Late Fusion')
ax2.plot(pose_error_b, ap50_nocolla, color='#4A4A4A',
         marker='None', linestyle='--', linewidth=3, label='No Collaboration')

ax2.set_xlabel('Pose Error (Std)')
ax2.set_ylabel('AP50')
ax2.grid(True, linestyle='-', alpha=0.8)
ax2.set_ylim(0.55, 0.88)
ax2.set_xticks([0.0, 0.2, 0.4, 0.6])
ax2.set_xlim(-0.05, 0.65)

# ---------------------- 6. 全局图例（与图距离更大，内部更紧凑） ----------------------
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # 去重

fig.legend(by_label.values(), by_label.keys(),
           loc='upper center',
           bbox_to_anchor=(0.5, 0.98),  # 仍然在顶部，中间
           ncol=4,
           frameon=False,
           fontsize=20,
           columnspacing=1.5,   # ← 列之间更紧凑
           handlelength=1.8,    # ← 线条短一点
           handletextpad=0.8)   # ← 线条和文字之间更近

# ---------------------- 7. 调整布局（加大图例与图之间距离） ----------------------
plt.subplots_adjust(
    top=0.84,   # ← 从 0.86 减到 0.80，子图整体往下，让图例和图之间空一点
    bottom=0.2,
    wspace=0.3,
)

# ---------------------- 8. 保存与显示 ----------------------
plt.savefig('pose_error.png',
            dpi=600, bbox_inches='tight', facecolor='white')

# plt.show()
