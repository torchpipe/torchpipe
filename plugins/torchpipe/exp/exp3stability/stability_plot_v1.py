import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import AutoLocator
from matplotlib.backends.backend_pdf import PdfPages


def parse_log_file(file_path):
    # 解析函数保持不变
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
                    '50': float(groups[7]),
                    '90': float(groups[8]),
                    '99': float(groups[9]),
                    '99.9': float(groups[10]),
                    '99.99': float(groups[11]),
                    '99.999': float(groups[12]),
                    'latency_avg': float(groups[13]),
                    'cpu_usage': float(groups[18]),
                    'gpu_usage': float(groups[19])
                }
                data.append(entry)
    return pd.DataFrame(data)


def plot_percentile_comparison(df, output_file="latency_percentiles.pdf"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Palatino", "New Century Schoolbook", "serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42
    })

    models = df['model'].unique().tolist()
    preprocessors = ['cpu', 'gpu']
    percentiles = [50, 90, 99, 99.9, 99.99, 99.999]
    percentile_labels = ['50', '90',
                         '99', '99.9', '99.99', '99.999']
    markers = ['o', 's', '^', 'D', 'v', '<']
    colors = {'cpu': '#1f77b4', 'gpu': '#d62728'}

    # 定义y轴位置（百分位点位置固定）
    y_positions = np.arange(len(percentile_labels))

    # 计算每个子图的宽度比例（基于延迟范围）
    width_ratios = []
    model_ranges = []

    for model in models:
        model_df = df[df['model'] == model]
        all_vals = model_df[percentile_labels].values

        # 计算数据范围（线性坐标）
        data_min = np.min(all_vals)
        data_max = np.max(all_vals)

        # 使用新的padding计算方式
        padding = (data_max - data_min) * 0.04 + 0.5
        x_min = max(0.1, data_min - padding)
        x_max = data_max + padding

        model_ranges.append((x_min, x_max))
        width_ratios.append(x_max - x_min)

    with PdfPages(output_file) as pdf:
        # 创建水平排列的子图（共享y轴）
        fig, axes = plt.subplots(
            1, len(models),
            figsize=(9, 4),  # 调整为横向半页尺寸
            gridspec_kw={
                'width_ratios': width_ratios,
                'wspace': 0.0  # 消除水平间距
            },
            sharey=True  # 共享y轴（百分位）
        )

        if len(models) == 1:
            axes = [axes]

        # 设置共享的y轴（百分位）
        for ax in axes:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(percentile_labels)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=1)

        for i, model in enumerate(models):
            ax = axes[i]
            model_df = df[df['model'] == model]
            model_short = model
            x_min, x_max = model_ranges[i]

            for pp in preprocessors:
                pp_df = model_df[model_df['preprocessor'] == pp]
                if not pp_df.empty:
                    values = pp_df.iloc[0][percentile_labels].values
                    # 翻转坐标：x=延迟值，y=百分位位置
                    ax.plot(values, y_positions,
                            marker=markers[i],
                            markersize=5,
                            linewidth=1.8,
                            color=colors[pp],
                            label=f'{pp.upper()} Preprocessor',
                            zorder=3)

            # 设置x轴范围和网格
            ax.set_xlim(x_min, x_max)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5, zorder=1)

            # 强制x轴刻度为整数
            ax.xaxis.set_major_locator(AutoLocator())
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

            # 添加模型标识
            ax.set_title(f'({chr(97+i)}) {model_short}', fontsize=10, pad=8)

            # 仅最左侧子图显示y轴标签
            if i == 0:
                ax.set_ylabel('Percentile', labelpad=8)

            # 仅底部子图显示x轴标签（实际所有子图都有x轴）
            ax.set_xlabel('Latency (ms)', fontsize=9,
                          labelpad=6) if i == len(models)//2 else None

        # 统一图例（放置在底部中央）
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='lower center',
                   ncol=2,
                   bbox_to_anchor=(0.5, 0.01),
                   frameon=True,
                   framealpha=0.9)

        # 调整布局
        plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.0,
                         rect=[0.03, 0.08, 0.97, 0.95])

        pdf.savefig(fig)
        plt.close()


if __name__ == "__main__":
    log_file = "../stability.log"
    output_pdf = "latency_percentiles_horizontal.pdf"
    df = parse_log_file(log_file)
    plot_percentile_comparison(df, output_pdf)
    print(f"分析完成! 图表已保存至: {output_pdf}")
