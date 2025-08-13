import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm


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
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,  # 减小全局字体大小
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42
    })

    model_names = {
        "mobilenetv2_100": "MobileNetV2",
        "vit_base_patch16_siglip_224": "ViT-Base",
        "resnet101": "ResNet101"
    }

    models_order = {
        'left': ['mobilenetv2_100'],
        'right_top': ['vit_base_patch16_siglip_224'],
        'right_bottom': ['resnet101']
    }

    preprocessors = ['cpu', 'gpu']
    percentiles = [50, 90, 99, 99.9, 99.99, 99.999]
    percentile_labels = ['50', '90', '99', '99.9', '99.99', '99.999']

    line_styles = {'cpu': '-', 'gpu': '--'}
    markers = {'cpu': 'o', 'gpu': 's'}
    colors = {'cpu': '#1f77b4', 'gpu': '#d62728'}

    model_ranges = {}
    height_ratios = {}

    for position, models in models_order.items():
        for model in models:
            model_df = df[df['model'] == model]
            all_vals = model_df[percentile_labels].values
            data_min = np.min(all_vals)
            data_max = np.max(all_vals)
            padding = (data_max - data_min) * 0.05 + 0.5
            y_min = max(0.1, data_min - padding)
            y_max = data_max + padding
            model_ranges[model] = (y_min, y_max)
            height_ratios[model] = y_max - y_min

    right_models = models_order['right_top'] + models_order['right_bottom']
    right_height_ratio = [height_ratios[model] for model in right_models]

    with PdfPages(output_file) as pdf:
        # 使用更紧凑的图形尺寸
        fig = plt.figure(figsize=(10, 7))

        # 创建网格布局，减少间距
        gs = plt.GridSpec(
            2, 2,
            figure=fig,
            width_ratios=[1, 1],
            height_ratios=right_height_ratio,
            hspace=0.30,  
            wspace=0.14   # 减少水平间距
        )

        ax_left = fig.add_subplot(gs[:, 0])
        ax_right_top = fig.add_subplot(gs[0, 1])
        ax_right_bottom = fig.add_subplot(gs[1, 1])

        axes = {
            'left': ax_left,
            'right_top': ax_right_top,
            'right_bottom': ax_right_bottom
        }

        # 绘制每个子图
        for position, ax in axes.items():
            model = models_order[position][0]
            model_df = df[df['model'] == model]
            model_short = model_names.get(model, model)
            y_min, y_max = model_ranges[model]

            for pp in preprocessors:
                pp_df = model_df[model_df['preprocessor'] == pp]
                if not pp_df.empty:
                    values = pp_df.iloc[0][percentile_labels].values
                    ax.plot(percentile_labels, values,
                            marker=markers[pp],
                            markersize=5,  # 减小标记大小
                            linestyle=line_styles[pp],
                            color=colors[pp],
                            label=f'{pp.upper()} Preprocessor',
                            zorder=3,
                            markeredgewidth=1.0)

            # 优化并发数文本位置和大小
            num_clients = model_df.iloc[0]['num_clients']
            ax.text(0.95, 0.08,  # 修改坐标：x=0.98（右侧），y=0.02（底部）
                    f"Request Concurrency: {num_clients}",
                    transform=ax.transAxes,  # 使用轴坐标系（0-1范围）
                    fontsize=14,  # 可进一步减小字体以适应角落
                    color='#444444',
                    horizontalalignment='right',  # 文本右边界对齐坐标点
                    verticalalignment='bottom',   # 文本底部对齐坐标点
                    bbox=dict(
                        boxstyle='round,pad=0.3',  # 减小内边距
                        facecolor='white',
                        alpha=0.75,
                        edgecolor='#bbbbbb',
                        linewidth=0.7
                    ))

            ax.grid(True, which="major", linestyle='--', alpha=0.7, zorder=1)
            ax.set_ylim(y_min, y_max)

            # 科学计数法处理
            if y_max > 1000:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(8)  # 减小偏移量字体大小

            # 设置更紧凑的子图标题
            ax.set_title(f'({chr(97+list(axes.keys()).index(position))}) {model_short}',
                         fontsize=15,  # 减小标题字体大小
                         fontweight='bold',
                         pad=10,  # 减小标题与子图的间距
                         y=1  # 降低标题位置
                         )

        # 优化坐标轴标签位置 - 使其更靠近图表
        fig.text(0.5, 0.035, 'Percentile', ha='center',
                 fontsize=15)  # , fontweight='bold'
        fig.text(0.035, 0.5, 'Latency (ms)', va='center',
                 rotation='vertical', fontsize=15)

        # 创建紧凑的图例
        handles, labels = ax_left.get_legend_handles_labels()
        legend = fig.legend(handles, labels,
                            loc='upper center',
                            ncol=2,
                            frameon=True,
                            framealpha=0.95,
                            edgecolor='#333333',
                            facecolor='white',
                            bbox_to_anchor=(0.5, 0.02),  # 提高图例位置
                            title_fontsize=15,
                            borderpad=0.5,  # 减小图例内边距
                            handlelength=1.5)  # 减小图例句柄长度
        legend.get_frame().set_linewidth(0.8)

        # 紧凑布局调整
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
        fig.subplots_adjust(
            bottom=0.12,  # 减少底部空间
            left=0.10,    # 减少左侧空间
            right=0.96,   # 减少右侧空间
            top=0.94      # 减少顶部空间
        )

        pdf.savefig(fig, dpi=600)
        plt.close()


if __name__ == "__main__":
    log_file = "../stability.log"
    output_pdf = "latency_percentiles.pdf"

    df = parse_log_file(log_file)
    plot_percentile_comparison(df, output_pdf)

    print(f"优化后的图表已保存至: {output_pdf}")
