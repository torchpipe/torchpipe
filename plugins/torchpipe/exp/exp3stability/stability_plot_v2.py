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
    # 设置学术出版级别的绘图参数
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
    markers = ['o', 's', '^', 'D', 'v', '<']  # 不同的标记样式

    # 创建颜色映射 - 更专业的学术配色
    colors = {
        'cpu': '#1f77b4',  # 蓝色
        'gpu': '#d62728'   # 红色
    }

    with PdfPages(output_file) as pdf:
        # 创建3个子图，水平排列
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

        # 设置全局标题
        fig.suptitle('Latency Percentile Comparison by Model and Preprocessor Type',
                     fontsize=12, y=0.98)

        # 确定Y轴范围
        all_latencies = df[percentile_labels].values.flatten()
        y_min = max(1, 0.8 * np.min(all_latencies))
        y_max = min(100, 1.2 * np.max(all_latencies))

        for i, model in enumerate(models):
            ax = axes[i]
            model_df = df[df['model'] == model]

            # 获取模型简称用于标题
            model_short = model.split('_')[0] if '_' in model else model

            for pp in preprocessors:
                pp_df = model_df[model_df['preprocessor'] == pp]
                if not pp_df.empty:
                    values = pp_df.iloc[0][percentile_labels].values

                    # 使用不同的标记和颜色
                    line = ax.semilogy(percentile_labels, values,
                                       marker=markers[i],
                                       markersize=5,
                                       linewidth=1.8,
                                       color=colors[pp],
                                       label=f'{pp.upper()} Preprocessor',
                                       zorder=3)

                    # 添加数据标签（仅关键点）
                    for j, v in enumerate(values):
                        if j == 0 or j == len(values)-1 or v == max(values):
                            ax.annotate(f'{v:.2f}ms',
                                        (j, v),
                                        textcoords="offset points",
                                        xytext=(0, 10 if j % 2 == 0 else -15),
                                        ha='center',
                                        fontsize=7)

            # 设置图表元素
            ax.set_title(f'{model_short}', fontsize=11, pad=10)
            ax.set_xlabel('Percentile', labelpad=8)
            if i == 0:
                ax.set_ylabel('Latency (ms)', labelpad=8)

            # 设置网格和刻度
            ax.grid(True, which="both", linestyle='--', alpha=0.5, zorder=1)
            ax.set_ylim(y_min, y_max)

            # 设置对数刻度的格式
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.tick_params(axis='x', rotation=45, pad=2)
            ax.tick_params(axis='y', which='both', pad=2)

            # 添加图例
            if i == 2:
                ax.legend(loc='upper left', framealpha=0.9)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(wspace=0.15, hspace=0.4)

        # 保存到PDF
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
