import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages

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

# 绘制稳定性图表（仅第一个图）

def plot_percentile_comparison(df, output_file="percentile_comparison.pdf"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "New Century Schoolbook"],
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

    models = df['model'].unique()
    preprocessors = ['cpu', 'gpu']
    percentiles = [50, 90, 99, 99.9, 99.99, 99.999]
    percentile_labels = ['TP50', 'TP90',
                         'TP99', 'TP99.9', 'TP99.99', 'TP99.999']
    markers = ['o', 's', '^', 'D', 'v', '<']
    colors = {'cpu': '#1f77b4', 'gpu': '#d62728'}

    # 计算每个模型的高度比例
    height_ratios = []
    model_ranges = []

    for model in models:
        model_df = df[df['model'] == model]
        all_vals = model_df[percentile_labels].values

        if "resnet" in model.lower():
            y_min, y_max = 6, 9
        else:
            data_min = np.min(all_vals)
            data_max = np.max(all_vals)
            padding = (data_max - data_min) * 0.05
            y_min = max(1, data_min - padding)
            y_max = data_max + padding

        model_ranges.append((y_min, y_max))
        height_ratios.append(y_max - y_min)

    with PdfPages(output_file) as pdf:
        # 创建具有动态高度比例的子图
        fig, axes = plt.subplots(
            len(models), 1,
            figsize=(6, 9),
            gridspec_kw={
                'height_ratios': height_ratios,
                'hspace': 0.0  # 完全消除垂直间距
            }
        )

        if len(models) == 1:
            axes = [axes]

        for i, model in enumerate(models):
            ax = axes[i]
            model_df = df[df['model'] == model]
            model_short = model.split('_')[0] if '_' in model else model
            y_min, y_max = model_ranges[i]

            for pp in preprocessors:
                pp_df = model_df[model_df['preprocessor'] == pp]
                if not pp_df.empty:
                    values = pp_df.iloc[0][percentile_labels].values
                    ax.semilogy(percentile_labels, values,
                                marker=markers[i],
                                markersize=5,
                                linewidth=1.8,
                                color=colors[pp],
                                label=f'{pp.upper()} Preprocessor',
                                zorder=3)

            # 设置网格和坐标轴
            ax.grid(True, which="both", linestyle='--', alpha=0.5, zorder=1)
            ax.set_ylim(y_min, y_max)

            # 设置整数格式的坐标轴
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.tick_params(axis='y', which='minor', length=0)

            # 强制y轴刻度为整数
            ax.yaxis.set_major_locator(plt.MaxNLocator(
                integer=True,
                steps=[1, 2, 5, 10],
                nbins=min(5, int(y_max - y_min) + 1)
            ))

            # 仅底部子图显示x轴标签
            if i == len(models) - 1:
                ax.set_xlabel('Percentile', labelpad=8)
            else:
                ax.set_xticklabels([])

            # 添加模型标识（使用相对位置）
            ax.text(0.02, 0.95, f'({chr(97+i)}) {model_short}',
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

        # 共享y轴标签（精确居中）
        fig.text(0.04, 0.5, 'Latency (ms)', va='center',
                 rotation='vertical', fontsize=10)

        # 统一图例（使用紧凑布局）
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.9)

        # 紧凑布局（完全消除所有边距）
        plt.tight_layout(pad=0.1, h_pad=0.0, w_pad=0.0,
                         rect=[0.04, 0.04, 0.99, 0.97])

        pdf.savefig(fig)
        plt.close()


# 主程序
if __name__ == "__main__":
    # 替换为实际文件路径
    log_file = "../stability.log"
    output_pdf = "latency_percentiles.pdf"

    # 解析日志并生成图表
    df = parse_log_file(log_file)
    plot_percentile_comparison(df, output_pdf)

    print(f"分析完成! 图表已保存至: {output_pdf}")
