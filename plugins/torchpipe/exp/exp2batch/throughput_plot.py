import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# 设置全局字体和样式
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'figure.figsize': (10, 6)
})

# 从CSV文件读取数据
df = pd.read_csv("benchmark_results.csv")

# 计算每个模型在batch_size=64时的基准吞吐量
base_throughputs = {}
for model in df['model'].unique():
    base = df[(df['model'] == model) & (
        df['batch_size'] == 64)]['throughput'].values
    if len(base) > 0:
        base_throughputs[model] = base[0]

# 计算归一化吞吐量
df['normalized_throughput'] = df.apply(
    lambda row: row['throughput'] / base_throughputs[row['model']]
    if row['model'] in base_throughputs else 1.0,
    axis=1
)

# 过滤掉batch_size=64的数据点
df = df[df['batch_size'] <= 16]

# 自动生成模型显示名称（移除.py扩展名，保持原始大小写）


def get_display_name(orig_name):
    if orig_name.endswith('.py'):
        target = orig_name[:-3]
        if target == 'seg':
            target = 'Segmentation Postprocessing'
        elif target == 'jpg_decode':
            target = 'JPEG Decoding'
        return target, True
    return orig_name, False


# 为每个模型分配唯一标识符
models = df['model'].unique().tolist()
models.sort(key=lambda model: model.endswith('.py'))
# models.sort(key=lambda model: len(get_display_name(model)))

model_ids = {model: idx for idx, model in enumerate(models)}
df['model_id'] = df['model'].map(model_ids)

# ====== 颜色方案修改：深色系用于模型，浅色系用于非模型 ======
# 创建深色和浅色调色板
dark_palette = plt.get_cmap('tab10')(np.linspace(0, 1, 10))  # 深色调色板
light_palette = plt.get_cmap('Pastel1')(np.linspace(0, 1, 10))  # 浅色调色板

# 合并调色板并创建颜色映射
combined_colors = np.vstack((dark_palette, light_palette))
cmap = LinearSegmentedColormap.from_list('custom_cmap', combined_colors, N=20)

# 设置线型
line_styles = {
    'image_processing': '--',  # 虚线表示图像处理
    'model_inference': '-'     # 实线表示模型推理
}

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 存储所有线条对象用于图例
lines = []
model_legend_elements = []

# 绘制每个模型的曲线
for orig_name in models:
    display_name, is_image = get_display_name(orig_name)
    model_df = df[df['model'] == orig_name].sort_values('batch_size')

    # 确定任务类型和样式
    is_image_processing = orig_name.endswith('.py')
    task_type = 'image_processing' if is_image_processing else 'model_inference'

    # ====== 关键修改：根据任务类型选择颜色索引 ======
    # 模型推理使用深色系 (0-9)，图像处理使用浅色系 (10-19)
    color_idx = model_ids[orig_name]*2 % 10  # 确保在0-9范围内
    if task_type == 'model_inference':
        color = cmap(color_idx + 0)   # 使用深色
        markersize= 2
        linewidth = 1.8
    else:
        color = cmap(color_idx + 8)  # 使用浅色（索引偏移到浅色区域）
        markersize = 2
        linewidth = 2.6

    # 绘制曲线
    line = ax.plot(
        model_df['batch_size'],
        model_df['normalized_throughput'],
        marker='o',
        markersize=markersize,
        markeredgewidth=0.8,
        markerfacecolor=color,
        markeredgecolor='white',
        linestyle=line_styles[task_type],
        linewidth=linewidth,
        color=color,
        alpha=0.9,
        markevery=4  # 每个点都显示标记
    )
    lines.append(line[0])

    # 为模型图例创建元素
    model_legend_elements.append(Line2D(
        [0], [0],
        marker='s' if is_image else  'o',
        color=color,
        label=display_name,
        linestyle='',
        markersize=8
    ))

# 标注第一个达到75%性能的点
for orig_name in models:
    display_name = get_display_name(orig_name)
    model_df = df[df['model'] == orig_name].sort_values('batch_size')

    # 重新计算颜色（确保与绘图一致）
    is_image_processing = orig_name.endswith('.py')
    color_idx = model_ids[orig_name]*2 % 10
    color = cmap(color_idx + 8) if is_image_processing else cmap(color_idx)

    # 找到第一个达到75%性能的点
    for idx, row in model_df.iterrows():
        if row['normalized_throughput'] >= 0.75:
            # 添加关键点标记
            ax.scatter(
                row['batch_size'],
                row['normalized_throughput'],
                s=50,  
                facecolors=color,
                edgecolors='none',  # 关键修改：移除外边界
                linewidths=3,  # 确保无边界线
                zorder=8
            )
            break

# 添加75%性能参考线
right_offset = 3
ax.axhline(y=0.75, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.text(16.2-right_offset-0.5, 0.71, '75% Reference',
        ha='right', fontsize=12, color='gray')

# 设置坐标轴和标题
# ax.set_title(
#     'Normalized Offline Throughput Comparison (Relative to Batch Size 64)', pad=15)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Normalized Offline Throughput')
ax.set_xticks(range(1, 17-right_offset))
# ax.set_xlim(1, 16.5-right_offset)
# ax.set_xticks(range(1, 17-right_offset-1))
ax.set_xlim(1, 17-right_offset-1)
ax.set_ylim(0.1, 1.05)
ax.grid(True, linestyle='--', alpha=0.3)

# 创建任务类型图例
task_legend_elements = [
    Line2D([0], [0], color='gray', lw=2,
           label='Model Inference', linestyle='-'),
    Line2D([0], [0], color='gray', lw=2,
           label='Image Processing', linestyle='--'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, markeredgecolor='gray', label='First ≥75% Point')
]

# 添加任务类型图例在右上角
task_legend = ax.legend(
    handles=task_legend_elements,
    loc='upper left',
    framealpha=0.6,
    facecolor='white',
    # title='Task Types'
)

# 添加模型名称图例在右下角（分两列显示）
# model_legend_elements.sort(key=lambda model: len(model._label))
model_legend = ax.legend(
    handles=model_legend_elements,
    loc='lower right',
    framealpha=0.9,
    facecolor='white',
    ncol=2,
    # title='Models'
)

# 添加两个图例
ax.add_artist(task_legend)
ax.add_artist(model_legend)

# 添加图表说明
# plt.figtext(0.5, -0.05,
#             "Note: Throughput normalized relative to each model's performance at batch size 64. "
#             "Dashed lines represent image processing tasks (light colors), solid lines represent model inference tasks (dark colors). "
#             "Marked points indicate the first batch size achieving ≥75% performance.",
#             ha="center", fontsize=9)


plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('normalized_throughput_comparison.pdf', bbox_inches='tight')
