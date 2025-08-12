import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

# 设置全局字体 - 使用适合学术出版的字体设置
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "axes.formatter.useoffset": False,
    "axes.formatter.limits": [-5, 6],
    "mathtext.default": "regular"  # 使用常规数学字体
})

# 解析日志文件的函数


def parse_log_file(file_path):
    pattern = re.compile(
        r"model:(.*?),preprocessor:(.*?),tool's version:(.*?),num_clients:(.*?),"
        r"total_number:(.*?),throughput::qps:(.*?),throughput::avg:(.*?),"
        r"latency::TP50:(.*?),latency::TP90:(.*?),latency::TP99:(.*?),"
        r"latency::TP99\.9:(.*?),latency::TP99\.99:(.*?),latency::TP99\.999:(.*?),"
        r"latency::avg:(.*?),-50:(.*?),-20:(.*?),-10:(.*?),-1:(.*?),"
        r"cpu_usage:(.*?),gpu_usage:(.*?)$"
    )

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if match:
                groups = match.groups()
                entry = {
                    'model': groups[0],
                    'preprocessor': groups[1],
                    'tool_version': groups[2],
                    'num_clients': int(groups[3]),
                    'total_requests': int(groups[4]),
                    'throughput_qps': float(groups[5]),
                    'throughput_avg': float(groups[6]),
                    'TP50': float(groups[7]),
                    'TP90': float(groups[8]),
                    'TP99': float(groups[9]),
                    'TP99.9': float(groups[10]),
                    'TP99.99': float(groups[11]),
                    'TP99.999': float(groups[12]),
                    'latency_avg': float(groups[13]),
                    'cpu_usage': float(groups[18]),
                    'gpu_usage': float(groups[19])
                }
                data.append(entry)
    return pd.DataFrame(data)

# 绘制单栏尺寸的百分位比较图


def plot_single_column_percentiles(df, output_file="single_column_percentiles.pdf"):
    # 单栏尺寸 (3.5英寸宽)
    SINGLE_COL_WIDTH = 3.5  # 英寸
    fig_height = SINGLE_COL_WIDTH * 0.75  # 保持宽高比

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, fig_height))

    # 优化绘图区域
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

    models = df['model'].unique()
    preprocessors = ['cpu', 'gpu']
    percentile_labels = ['TP50', 'TP90',
                         'TP99', 'TP99.9', 'TP99.99', 'TP99.999']

    # 模型简称映射 - 学术论文中使用标准缩写
    model_short_names = {
        'resnet101': 'ResNet-101',
        'mobilenetv2_100': 'MobiNetV2',
        'vit_base_patch16_siglip_224': 'ViT-SigLIP'
    }

    # 预处理方式显示名称
    pp_names = {
        'cpu': 'CPU',
        'gpu': 'GPU'
    }

    # 创建颜色映射 - 使用更学术的调色板
    colors = {
        'resnet101': '#1f77b4',  # 蓝色
        'mobilenetv2_100': '#2ca02c',  # 绿色
        'vit_base_patch16_siglip_224': '#d62728'  # 红色
    }

    # 线型和标记样式
    linestyles = {
        'cpu': '-',
        'gpu': '--'
    }

    markers = {
        'cpu': 'o',
        'gpu': 's'
    }

    # 绘制所有数据线
    for model in models:
        model_df = df[df['model'] == model]
        model_short = model_short_names.get(model, model)

        for pp in preprocessors:
            pp_df = model_df[model_df['preprocessor'] == pp]
            if not pp_df.empty:
                values = pp_df.iloc[0][percentile_labels].values
                line_style = linestyles[pp]
                marker_style = markers[pp]

                # 绘制线条
                ax.plot(percentile_labels, values,
                        marker=marker_style,
                        markersize=4.5,
                        linewidth=1.2,
                        color=colors[model],
                        linestyle=line_style,
                        label=f"{model_short} ({pp_names[pp]})",
                        zorder=3,
                        markeredgewidth=0.5,
                        markeredgecolor='w')  # 白色边缘提高可读性

                # 标注尾部延迟值
                tail_value = values[-1]
                ax.annotate(f'{tail_value:.1f}',
                            (percentile_labels[-1], tail_value),
                            textcoords="offset points",
                            xytext=(5, 0),
                            ha='left',
                            va='center',
                            fontsize=6.5,
                            color=colors[model],
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc=(1, 1, 1, 0.7),
                                      ec=colors[model],
                                      lw=0.5))

    # 设置图表元素
    ax.set_title('Latency Percentile Comparison', fontsize=10, pad=10)
    ax.set_xlabel('Percentile', labelpad=6)
    ax.set_ylabel('Latency (ms)', labelpad=6)

    # 优化刻度标签
    ax.set_xticklabels([x.replace('TP', '') for x in percentile_labels], rotation=20,
                       ha='right', fontsize=7.5)

    # 设置Y轴范围
    all_latencies = df[percentile_labels].values.flatten()
    y_max = max(20, np.percentile(all_latencies, 99) * 1.25)
    ax.set_ylim(0, y_max)

    # 使用对数刻度更好地显示长尾分布
    # ax.set_yscale('log')
    # ax.set_ylim(1, y_max)

    # 添加网格线
    ax.grid(True, axis='y', linestyle=':', alpha=0.4, zorder=1)
    ax.grid(True, axis='x', linestyle=':', alpha=0.2, zorder=1)

    # 添加图例 - 紧凑排列在右上角
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc='upper right',
              frameon=True,
              framealpha=0.9,
              ncol=2,
              columnspacing=0.8,
              handletextpad=0.4,
              handlelength=2.0,
              fontsize=7.5)

    # 添加学术风格的边框
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    # 保存到PDF
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, dpi=600)

    plt.close()


# 主程序
if __name__ == "__main__":
    # 替换为实际文件路径
    log_file = "../stability.log"
    output_pdf = "single_column_percentiles.pdf"

    # 解析日志并生成图表
    df = parse_log_file(log_file)
    plot_single_column_percentiles(df, output_pdf)

    print(f"Analysis complete! Single-column chart saved to: {output_pdf}")
