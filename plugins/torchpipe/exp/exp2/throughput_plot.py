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
    'lines.linewidth': 1.8,
    'patch.edgecolor': 'black'
})

# 从CSV文件读取数据
df = pd.read_csv('benchmark_results.csv')

# 获取所有模型列表
models = df['model'].unique()
print(f"Found {len(models)} models: {', '.join(models)}")

# 计算每个模型的归一化吞吐量
for model in models:
    model_data = df[df['model'] == model]
    base_throughput = model_data[model_data['batch_size']
                                 == 32]['throughput'].values[0]
    df.loc[df['model'] == model, 'normalized_throughput'] = df[df['model']
                                                               == model]['throughput'] / base_throughput

# 提取batch_size小于32的数据点（排除基准点）
plot_df = df[df['batch_size'] < 32].sort_values(['model', 'batch_size'])

print(plot_df)
# 创建图形
plt.figure(figsize=(7, 5))

# 定义学术配色方案
PALETTE = sns.color_palette("colorblind", n_colors=len(models))
model_colors = dict(zip(models, PALETTE))

# 绘制75%阈值线
plt.axhline(y=0.75, color='#d62728', linestyle='--',
            linewidth=1.5, alpha=0.85, zorder=1,
            label='75% Throughput Threshold')

# 存储关键点信息用于图例
min_bs_points = []

# 为每个模型绘制曲线
for model in models:
    model_data = plot_df[plot_df['model'] == model]
    color = model_colors[model]

    # 绘制主曲线
    plt.plot(model_data['batch_size'], model_data['normalized_throughput'],
             marker='o', linestyle='-', color=color,
             markersize=6, markeredgecolor='white', markeredgewidth=0.8,
             zorder=3, label=model)

    # 找出并标记第一个达到75%性能的batch size
    valid_points = model_data[model_data['normalized_throughput'] >= 0.75]
    if not valid_points.empty:
        min_bs = valid_points['batch_size'].min()
        min_point = valid_points[valid_points['batch_size'] == min_bs].iloc[0]
        min_bs_points.append((model, min_bs))

        # 添加关键点标记
        plt.plot(min_bs, min_point['normalized_throughput'], 'o',
                 color='#2ca02c', markersize=8,
                 markeredgecolor='white', markeredgewidth=1.2, zorder=4)

        # 添加关键点标签
        plt.text(min_bs, min_point['normalized_throughput'] + 0.05,
                 f"BS={min_bs}",
                 ha='center', fontsize=8, color='#2ca02c',
                 weight='bold', bbox=dict(facecolor='white', alpha=0.8,
                                          edgecolor='none', boxstyle='round,pad=0.2'))

# 设置图表属性
plt.title('Normalized Offline Throughput vs Batch Size',
          fontsize=12, pad=12, weight='bold')
plt.xlabel('Batch Size', fontsize=11, labelpad=8)
plt.ylabel('Normalized Throughput (BS=32 = 1.0)',
           fontsize=11, labelpad=8)

# 设置坐标轴刻度
plt.xticks(np.arange(1, 17, 1))
plt.xlim(0.5, 16.5)
plt.yticks(np.arange(0.4, 1.21, 0.1))
plt.ylim(0.35, 1.15)

# 添加精细刻度
ax = plt.gca()
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

# 优化网格
plt.grid(True, linestyle='--', alpha=0.15, which='major')
plt.grid(True, linestyle=':', alpha=0.08, which='minor')

# 创建图例
handles, labels = plt.gca().get_legend_handles_labels()

# 添加关键点图例项
min_bs_legend = []
for model, min_bs in min_bs_points:
    min_bs_legend.append(f"{model} Min BS ({min_bs})")

# 合并图例
all_handles = handles
all_labels = labels + min_bs_legend

# 调整图例位置和样式
plt.legend(all_handles, all_labels,
           loc='lower right', frameon=True, framealpha=0.92,
           edgecolor='0.8', fancybox=False, handlelength=1.8,
           borderpad=0.6, handletextpad=0.8, ncol=1)

# 添加脚注说明
min_bs_str = ", ".join([f"{model}(BS={bs})" for model, bs in min_bs_points])
plt.figtext(0.5, 0.01,
            f'Fig: Normalized throughput across batch sizes. Minimum viable batch sizes: {min_bs_str}',
            ha='center', fontsize=8, style='italic', color='#555555')

# 调整布局并保存
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('multi_model_normalized_throughput.pdf')
plt.show()
