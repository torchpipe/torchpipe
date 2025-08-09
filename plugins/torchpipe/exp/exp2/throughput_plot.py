import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator

# 设置学术论文风格的字体配置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.2,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.6,
    'patch.edgecolor': 'black'
})

# 从CSV文件读取数据
df = pd.read_csv('benchmark_results.csv')

# 获取batch_size=32时的基准吞吐量
base_throughput = df[df['batch_size'] == 32]['throughput'].values[0]

# 计算归一化吞吐量
df['normalized_throughput'] = df['throughput'] / base_throughput

# 提取batch_size小于32的数据点（排除基准点）
plot_df = df[df['batch_size'] < 32].sort_values('batch_size')
print(plot_df)
# 创建图形
plt.figure(figsize=(6, 4.5))

# 定义淡雅学术配色方案
PALETTE = {
    'baseline': '#7f7f7f',      # 中性灰
    'threshold': '#d62728',     # 暗红色
    'main_line': '#1f77b4',     # 淡蓝色
    'highlight': '#2ca02c',     # 淡绿色
    'data_label': '#333333'     # 深灰色
}

# 绘制基准线和阈值线
# plt.axhline(y=1.0, color=PALETTE['baseline'], linestyle='-',
#             linewidth=1.2, alpha=0.9, zorder=1,
#             label='Baseline (BS=32)')
plt.axhline(y=0.75, color=PALETTE['threshold'], linestyle='--',
            linewidth=1.5, alpha=0.85, zorder=1,
            label='75% Throughput Threshold')

# 绘制归一化吞吐量曲线
plt.plot(plot_df['batch_size'], plot_df['normalized_throughput'],
         marker='o', linestyle='-', color=PALETTE['main_line'],
         markersize=5.5, markeredgecolor='white', markeredgewidth=0.8,
         zorder=3, label='Normalized Throughput')

# 找出并标记第一个达到75%性能的batch size
valid_points = plot_df[plot_df['normalized_throughput'] >= 0.75]
if not valid_points.empty:
    min_bs = valid_points['batch_size'].min()
    min_point = valid_points[valid_points['batch_size'] == min_bs].iloc[0]

    # 添加带标注的关键点
    plt.plot(min_bs, min_point['normalized_throughput'], 'o',
             color=PALETTE['highlight'], markersize=7.5,
             markeredgecolor='white', markeredgewidth=0.8, zorder=4,
             label=f'Minimum Viable BS ({min_bs})')

# 添加数据标签
for i, row in plot_df.iterrows():
    offset = 0.03 if row['normalized_throughput'] < 0.85 else -0.04
    va = 'bottom' if row['normalized_throughput'] < 0.85 else 'top'

    # 特殊处理关键点
    if 'min_point' in locals() and row['batch_size'] == min_bs:
        plt.text(row['batch_size'], row['normalized_throughput'] + offset,
                 f"{row['normalized_throughput']:.2f}",
                 ha='center', fontsize=8, color=PALETTE['highlight'],
                 weight='bold', va=va, bbox=dict(facecolor='white', alpha=0.8,
                 edgecolor='none', boxstyle='round,pad=0.2'))
    else:
        plt.text(row['batch_size'], row['normalized_throughput'] + offset,
                 f"{row['normalized_throughput']:.2f}",
                 ha='center', fontsize=8, color=PALETTE['data_label'],
                 va=va, alpha=0.9)

# 设置图表属性
plt.title('Normalized Offline Throughput vs Batch Size',
          fontsize=11, pad=12, weight='bold')
plt.xlabel('Batch Size', fontsize=10, labelpad=8)
plt.ylabel('Normalized Throughput\n(BS=32 = 1.0)',
           fontsize=10, labelpad=8)

# 设置坐标轴刻度
plt.xticks(np.arange(1, 17, 2))  # 只显示奇数刻度
plt.xlim(0.5, 16.5)
plt.yticks(np.arange(0.4, 1.41, 0.1))
plt.ylim(0.35, 1)

# 添加精细刻度
ax = plt.gca()
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

# 优化网格
plt.grid(True, linestyle='--', alpha=0.15, which='major')
plt.grid(True, linestyle=':', alpha=0.08, which='minor')

# 优化图例位置和样式
plt.legend(loc='lower right', frameon=True, framealpha=0.92,
           edgecolor='0.8', fancybox=False, handlelength=1.5,
           borderpad=0.4, handletextpad=0.5)

# 添加脚注说明
plt.figtext(0.5, 0.01,
            'Fig. 1: Normalized throughput across batch sizes. Minimum viable batch size (BS={}) maintains ≥75% of BS=32 throughput.'
            .format(min_bs),
            ha='center', fontsize=8, style='italic', color='#555555')

# 调整布局并保存
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('normalized_offline_throughput.pdf')
# plt.savefig('normalized_offline_throughput.png')  # 同时保存PNG版本
plt.show()
