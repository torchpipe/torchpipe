import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import AutoLocator
from matplotlib.backends.backend_pdf import PdfPages


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
    data.sort(key=lambda entry: 'resnet' not in entry['model'])
    return pd.DataFrame(data)


def plot_percentile_comparison(df, output_file="latency_percentiles.pdf"):
    # 优化线条粗细设置
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Palatino", "Bookman", "serif"],
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 1.8,  # 减小线条宽度，使更协调
        "lines.markersize": 8,   # 减小标记尺寸
        "axes.linewidth": 1.0,   # 减细坐标轴线
        "grid.linewidth": 0.4,   # 减细网格线
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42
    })

    models = df['model'].unique().tolist()
    preprocessors = ['cpu', 'gpu']
    percentiles = [50, 90, 99, 99.9, 99.99, 99.999]
    percentile_labels = ['50', '90', '99', '99.9', '99.99', '99.999']
    # markers = ['o', 's', '^', 'D', 'v', '<']
    markers = ['']*10
    colors = {'cpu': '#1f77b4', 'gpu': '#d62728'}

    # 添加线条样式区分
    line_styles = {'cpu': '-', 'gpu': '--'}  # 实线 vs 虚线

    height_ratios = []
    model_ranges = []

    for model in models:
        model_df = df[df['model'] == model]
        all_vals = model_df[percentile_labels].values
        data_min = np.min(all_vals)
        data_max = np.max(all_vals)
        padding = (data_max - data_min) * 0.04 + 0.5
        y_min = max(0.1, data_min - padding)
        y_max = data_max + padding
        model_ranges.append((y_min, y_max))
        height_ratios.append(y_max - y_min)

    with PdfPages(output_file) as pdf:
        fig = plt.figure(figsize=(9, 8.5))

        gs = plt.GridSpec(len(models), 1, figure=fig,
                          height_ratios=height_ratios, hspace=0.0)
        axes = [fig.add_subplot(gs[i]) for i in range(len(models))]

        for i, model in enumerate(models):
            ax = axes[i]
            model_df = df[df['model'] == model]
            model_short = model
            y_min, y_max = model_ranges[i]

            for pp in preprocessors:
                pp_df = model_df[model_df['preprocessor'] == pp]
                if not pp_df.empty:
                    values = pp_df.iloc[0][percentile_labels].values
                    # 添加线条样式区分，减小标记尺寸
                    ax.plot(percentile_labels, values,
                            marker=markers[i],
                            markersize=7,      # 减小标记尺寸
                            linewidth=1.8,     # 减小线条宽度
                            linestyle=line_styles[pp],  # 添加线条样式
                            color=colors[pp],
                            label=f'{pp.upper()} Preprocessor',
                            zorder=3)

            # 优化标注框样式
            num_clients = model_df.iloc[0]['num_clients']
            ax.text(0.98, 0.15, f"Request Concurrency: {num_clients}",
                    transform=ax.transAxes,
                    fontsize=11,
                    color='#555555',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white',
                              alpha=0.8,
                              edgecolor='#dddddd',  # 更浅的边框颜色
                              linewidth=0.8))       # 减细边框

            # 使用更柔和的网格线
            ax.grid(True, which="both", linestyle=':', alpha=0.5, zorder=1)
            ax.set_ylim(y_min, y_max)

            # 减细刻度线
            ax.tick_params(axis='y', which='both', length=4, width=0.8)
            ax.tick_params(axis='x', which='both', length=4, width=0.8)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

            if i == len(models) - 1:
                ax.set_xlabel('Percentile', fontsize=16, labelpad=12)
            else:
                ax.set_xticklabels([])

            # 优化模型标识框
            ax.text(0.02, 0.85, f'({chr(97+i)}) {model_short}',
                    transform=ax.transAxes,
                    fontsize=16,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4',
                              fc='white',
                              ec='#888888',  # 更柔和的边框颜色
                              alpha=0.9,
                              linewidth=0.8))  # 减细边框

        fig.text(0.04, 0.5, 'Latency (ms)', va='center',
                 rotation='vertical', fontsize=16)

        # 优化图例位置和样式
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', ncol=1,
                   fontsize=13,  # 减小图例字体
                   bbox_to_anchor=(0.98, 0.03),
                   frameon=True,
                   framealpha=0.9,
                   edgecolor='#cccccc',  # 更柔和的边框颜色
                   title='',
                   title_fontsize=12)

        # 调整布局
        plt.tight_layout(pad=0.1, h_pad=0.0, w_pad=0.0,
                         rect=[0.06, 0.06, 0.95, 0.97])

        pdf.savefig(fig)
        plt.close()



if __name__ == "__main__":
    log_file = "../stability.log"
    output_pdf = "latency_percentiles.pdf"  # 更新输出文件名

    df = parse_log_file(log_file)
    plot_percentile_comparison(df, output_pdf)

    print(f"优化图表已保存至: {output_pdf}")
