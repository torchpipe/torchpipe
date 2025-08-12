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

# 绘制稳定性图表（垂直紧密叠加）


def plot_percentile_comparison(df, output_file="percentile_comparison.pdf"):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "New Century Schoolbook"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.3,
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

    # 为每个模型定义优化的y轴范围
    model_ranges = {
        "resnet101": (6, 9),      # 显著缩小范围
        "yolov8": (10, 40),       # 中等范围
        "bert-base": (20, 100)    # 较大范围
    }

    with PdfPages(output_file) as pdf:
        # 创建3个垂直排列的子图（3行1列）
        fig, axes = plt.subplots(3, 1, figsize=(4.5, 7))  # 更紧凑的尺寸

        # 获取全局y轴范围（基于所有数据）
        all_latencies = df[percentile_labels].values.flatten()

        for i, model in enumerate(models):
            ax = axes[i]
            model_df = df[df['model'] == model]
            model_short = model.split('_')[0] if '_' in model else model

            # 获取该模型的优化y轴范围
            y_min, y_max = model_ranges.get(
                model, (0.8*np.min(all_latencies), 1.2*np.max(all_latencies)))

            for pp in preprocessors:
                pp_df = model_df[model_df['preprocessor'] == pp]
                if not pp_df.empty:
                    values = pp_df.iloc[0][percentile_labels].values
                    line = ax.semilogy(percentile_labels, values,
                                       marker=markers[i],
                                       markersize=4.5,  # 更小的标记
                                       linewidth=1.5,
                                       color=colors[pp],
                                       label=f'{pp.upper()} Preprocessor',
                                       zorder=3)

            # 设置网格和刻度
            ax.grid(True, which="both", linestyle='--', alpha=0.4, zorder=1)
            ax.set_ylim(y_min, y_max)

            # 使用线性刻度（非对数）因为范围较小
            ax.set_yscale('linear')

            # 设置y轴刻度位置和标签
            ax.yaxis.set_major_locator(plt.MultipleLocator((y_max - y_min)/4))
            ax.yaxis.set_minor_locator(plt.MultipleLocator((y_max - y_min)/20))

            # 只在底部子图显示x轴标签
            if i == 2:
                ax.set_xlabel('Percentile', labelpad=6)
            else:
                ax.set_xticklabels([])  # 移除上方子图的x轴刻度标签

            # 添加模型标识（替代标题）
            ax.text(0.02, 0.95, f'{model_short}',
                    transform=ax.transAxes,
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment='top')

        # 设置共享的y轴标签（居中）
        fig.text(0.04, 0.5, 'Latency (ms)', va='center',
                 rotation='vertical', fontsize=9)

        # 添加统一图例（在底部）
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2,
                   bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.8)

        # 调整子图间距（垂直无间隙）
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
        plt.subplots_adjust(hspace=0.0)  # 垂直方向无间隙

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
