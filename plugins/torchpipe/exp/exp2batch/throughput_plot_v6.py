import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# === 增强的全局设置 ===
matplotlib.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 600,
    'figure.figsize': (10, 6.5),
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
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

# === 优化视觉设计 ===
# 使用更协调的配色方案（基于Tableau调色板）
optimized_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

# 定义多种标记符号
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', 'd', '*', 'X', 'P']

# 优化线型设置 - 增加虚线密度使其更明显
line_styles = {
    'image_processing': (0, (3, 2)),  # 更密集的虚线
    'model_inference': (0, (1, 0))    # 实线保持不变
}

# 创建图形
fig, ax = plt.subplots(figsize=(8, 5.5))

# 存储图例元素
model_legend_elements = []
task_legend_elements = []

# 绘制曲线
for orig_name in models:
    display_name, is_image = get_display_name(orig_name)
    model_df = df[df['model'] == orig_name].sort_values('batch_size')

    # 确定样式
    task_type = 'image_processing' if is_image else 'model_inference'
    color_idx = model_ids[orig_name] % len(optimized_colors)
    base_color = optimized_colors[color_idx]

    # 分配视觉属性
    line_style = line_styles[task_type]
    marker = markers[color_idx % len(markers)]
    line_width = 2.2 if is_image else 2.3  # 略微增加线宽

    marker_size = 7 if is_image else 8
    marker_edge_width = 1.5  # 略微减小标记边缘宽度

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
        alpha=0.95,  # 略微增加不透明度
        zorder=5 if is_image else 5,
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
        markeredgewidth=1.2,  # 减小图例标记边缘宽度
        linestyle='',
        label=display_name
    ))

    # 标记75%性能点
    for idx, row in model_df.iterrows():
        if row['normalized_throughput'] >= 0.75:
            # 先绘制一个稍大的背景星形
            ax.plot(
                row['batch_size'],
                row['normalized_throughput'],
                marker='*',
                markersize=18,
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
                markersize=16,
                markeredgewidth=1,
                markerfacecolor='#FFD700',
                markeredgecolor='#333333',
                zorder=20
            )
            break

# === 新增：计算参考点性能对比 ===
reference_point_results = []

for orig_name in models:
    model_df = df[df['model'] == orig_name].sort_values('batch_size')

    # 找到第一个达到75%性能的点
    ref_point = None
    for idx, row in model_df.iterrows():
        if row['normalized_throughput'] >= 0.75:
            ref_point = row
            break

    if ref_point is not None:
        ref_batch_size = ref_point['batch_size']
        ref_throughput = ref_point['throughput']

        # 获取batch_size=1的数据
        bs1_data = model_df[model_df['batch_size'] == 1]
        if not bs1_data.empty:
            bs1_throughput = bs1_data['throughput'].values[0]

            # 计算吞吐量提升倍数
            throughput_ratio = ref_throughput / bs1_throughput

            # 计算延迟降低倍数（吞吐量的倒数）
            latency_ratio = (bs1_throughput) / (ref_throughput/ref_batch_size)

            display_name, is_image = get_display_name(orig_name)

            reference_point_results.append({
                'model': display_name,
                'reference_batch_size': ref_batch_size,
                'throughput_ratio': throughput_ratio,
                'latency_ratio': latency_ratio,
                'is_image_processing': is_image
            })

print("=" * 80)
print("参考点性能对比分析 (与batch_size=1相比)")
print("=" * 80)
print(f"{'模型':<25} {'参考点Batch Size':<18} {'吞吐倍数':<15} {'延迟倍数':<15}")
print("-" * 80)

for result in reference_point_results:
    print(f"{result['model']:<25} {result['reference_batch_size']:<18} "
          f"{result['throughput_ratio']:<15.2f} {result['latency_ratio']:<15.2f}")

avg_throughput_ratio = np.mean([r['throughput_ratio']
                               for r in reference_point_results])
avg_latency_ratio = np.mean([r['latency_ratio']
                            for r in reference_point_results])

print("-" * 80)
print(f"{'平均值':<25} {'-':<18} {avg_throughput_ratio:<15.2f} {avg_latency_ratio:<15.2f}")
print("=" * 80)

# 优化参考线样式
ax.axhline(y=0.75, color='#555555', linestyle=':', linewidth=2.2, alpha=0.85)

ax.text(0.91, 0.594, '75% Reference Line',
        transform=ax.transAxes, ha='right', fontsize=11.5,  # 略微减小字体
        color='#555555', style='italic')

ax.set_xlabel('Batch Size', fontsize=15, labelpad=10)
ax.set_ylabel('Normalized Offline Throughput', fontsize=15, labelpad=10)
ax.set_xticks(range(1, mbs+1))
ax.set_xlim(0.8, mbs+0.2)
ax.set_ylim(0.1, 1.1)
ax.set_yticks(np.arange(0.2, 1.1, 0.2))
ax.grid(True, which='major', linestyle='--', alpha=0.4)

task_legend_elements.append(
    Line2D([0], [0], color='black', lw=2.5, linestyle=line_styles['model_inference'], label='Model Inference'))
task_legend_elements.append(Line2D(
    [0], [0], color='black', lw=2.5, linestyle=line_styles['image_processing'], label='Image Processing'))
task_legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
                                   markersize=16, markeredgecolor='#333333', label='First ≥75% Point'))

task_legend = ax.legend(
    handles=task_legend_elements,
    loc='upper left',
    framealpha=0.75,
    facecolor='white',
    frameon=True,
    edgecolor='lightgray',
    title='',
    title_fontsize=13
)

model_legend = ax.legend(
    handles=model_legend_elements,
    loc='lower right',
    framealpha=0.95,
    facecolor='white',
    ncol=1,
    frameon=True,
    edgecolor='lightgray',
    title='',
    title_fontsize=13
)

ax.add_artist(model_legend)
ax.add_artist(task_legend)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig('normalized_throughput.pdf',
            bbox_inches='tight',
            dpi=600,
            pad_inches=0.1)
print(f'saved to normalized_throughput.pdf')
