import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# === 专业级视觉设置 ===
matplotlib.rcParams.update({
    'font.size': 14,
    'font.family': 'DejaVu Sans',  # 使用更现代的字体
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 600,
    'figure.figsize': (10, 7),
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
})

# 读取数据
df = pd.read_csv("benchmark_results.csv")

# 计算基准吞吐量
base_throughputs = {}
for model in df['model'].unique():
    base = df[(df['model'] == model) & (
        df['batch_size'] == 64)]['throughput'].values
    if len(base) > 0:
        base_throughputs[model] = base[0]

# 计算归一化吞吐量
df['normalized_throughput'] = df.apply(
    lambda row: row['throughput'] / base_throughputs[row['model']
                                                     ] if row['model'] in base_throughputs else 1.0,
    axis=1
)

# 过滤数据
mbs = 13
df = df[df['batch_size'] <= mbs]

# 模型显示名称处理


def get_display_name(orig_name):
    if orig_name.endswith('.py'):
        target = orig_name[:-3]
        if target == 'seg':
            return 'Segmentation Postprocessing', True
        if target == 'jpg_decode':
            return 'JPEG Decoding', True
        return target.replace('_', ' ').title(), True
    return orig_name, False


# 模型分组和排序
models = df['model'].unique().tolist()
models.sort(key=lambda model: (not model.endswith('.py'), model))
model_ids = {model: idx for idx, model in enumerate(models)}
df['model_id'] = df['model'].map(model_ids)

# === 精心设计的配色方案 ===
# 使用专业、和谐的配色方案
professional_colors = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A8EAE',
    '#5C80BC', '#4D5061', '#E3B505', '#95190C', '#610345',
    '#107E7D', '#044B7F', '#5C8001', '#F24236', '#5D5E60'
]

# 定义多种标记符号
markers = ['o', 's', '^', 'D', 'v', 'p', 'h',
           'd', 'X', 'P', '8', 'H', '*', '<', '>']

# 线型设置
line_styles = {
    'image_processing': (0, (5, 2)),  # 更优雅的虚线
    'model_inference': (0, (1, 0))    # 实线
}

# 创建图形和背景
fig, ax = plt.subplots(figsize=(10, 7))
# fig.patch.set_facecolor('#F8F9FA')  # 设置图形背景色
# ax.set_facecolor('#FFFFFF')  # 设置坐标轴区域背景色

# 存储图例元素
model_legend_elements = []
task_legend_elements = []

# 绘制曲线
for orig_name in models:
    display_name, is_image = get_display_name(orig_name)
    model_df = df[df['model'] == orig_name].sort_values('batch_size')

    # 确定样式
    task_type = 'image_processing' if is_image else 'model_inference'
    color_idx = model_ids[orig_name] % len(professional_colors)
    base_color = professional_colors[color_idx]

    # 分配视觉属性
    line_style = line_styles[task_type]
    marker = markers[color_idx % len(markers)]
    line_width = 2.8 if is_image else 2.3  # 细微的线宽差异

    # 标记属性
    marker_size = 7 if is_image else 6
    marker_edge_width = 1.8

    # 绘制曲线
    ax.plot(
        model_df['batch_size'],
        model_df['normalized_throughput'],
        marker=marker,
        markersize=marker_size,
        markeredgewidth=marker_edge_width,
        markeredgecolor='white',
        markerfacecolor=base_color,
        linestyle=line_style,
        linewidth=line_width,
        color=base_color,
        alpha=0.9,
        zorder=10 if is_image else 5,
        markevery=2
    )

    # 添加模型图例元素
    model_legend_elements.append(Line2D(
        [0], [0],
        marker=marker,
        color=base_color,
        markeredgecolor='white',
        markerfacecolor=base_color,
        markersize=10,
        markeredgewidth=1.5,
        linestyle='',
        label=display_name
    ))

    # 标记75%性能点 - 使用醒目的五角星
    for idx, row in model_df.iterrows():
        if row['normalized_throughput'] >= 0.75:
            # 先绘制一个稍大的背景星形
            ax.plot(
                row['batch_size'],
                row['normalized_throughput'],
                marker='*',
                markersize=30,
                markeredgewidth=0,
                markerfacecolor='#333333',
                markeredgecolor='#333333',
                zorder=19,
                alpha=0.6
            )
            # 绘制主星形
            ax.plot(
                row['batch_size'],
                row['normalized_throughput'],
                marker='*',
                markersize=26,
                markeredgewidth=1.5,
                markerfacecolor='#FFD700',
                markeredgecolor='#333333',
                zorder=20
            )
            break

# 添加75%参考线 - 更优雅的样式
ax.axhline(y=0.75, color='#666666', linestyle=':', linewidth=2.5, alpha=0.8)
ax.text(0.95, 0.66, '75% Reference Line',
        transform=ax.transAxes, ha='right', fontsize=11,
        color='#666666', style='italic')

# 坐标轴标签和刻度
ax.set_xlabel('Batch Size', fontsize=14, labelpad=10, fontweight='medium')
ax.set_ylabel('Normalized Offline Throughput', fontsize=14,
              labelpad=10, fontweight='medium')
ax.set_xticks(range(1, mbs+1))
ax.set_xlim(0.8, mbs+0.2)
ax.set_ylim(0.1, 1.02)
ax.set_yticks(np.arange(0.2, 1.1, 0.2))

# 网格样式优化
ax.grid(True, which='major', linestyle='--', alpha=0.4, color='#CCCCCC')

# 添加任务类型图例
task_legend_elements.append(
    Line2D([0], [0], color='#333333', lw=2.3, linestyle=line_styles['model_inference'], label='Model Inference'))
task_legend_elements.append(Line2D(
    [0], [0], color='#333333', lw=2.8, linestyle=line_styles['image_processing'], label='Image Processing'))
task_legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
                                   markersize=16, markeredgecolor='#333333', label='First ≥75% Point'))

# 任务类型图例
task_legend = ax.legend(
    handles=task_legend_elements,
    loc='upper left',
    framealpha=0.9,
    facecolor='white',
    frameon=True,
    edgecolor='#DDDDDD',
    # title='Task Types',
    title_fontsize=12,
    bbox_to_anchor=(0.02, 0.98)
)

# 模型图例
model_legend = ax.legend(
    handles=model_legend_elements,
    loc='lower right',
    framealpha=0.95,
    facecolor='white',
    ncol=1,
    frameon=True,
    edgecolor='#DDDDDD',
    # title='Models',
    title_fontsize=12,
    bbox_to_anchor=(0.98, 0.02)
)

ax.add_artist(model_legend)
ax.add_artist(task_legend)

# 添加标题
# plt.title('Normalized Throughput vs. Batch Size',
#           fontsize=16, pad=20, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存高质量图像
plt.savefig('normalized_throughput.pdf',
            bbox_inches='tight',
            dpi=600,
            pad_inches=0.1,
            facecolor=fig.get_facecolor(),
            edgecolor='none')

plt.savefig('normalized_throughput.png',
            bbox_inches='tight',
            dpi=600,
            pad_inches=0.1,
            facecolor=fig.get_facecolor(),
            edgecolor='none')

print('Saved to normalized_throughput.pdf and normalized_throughput.png')
