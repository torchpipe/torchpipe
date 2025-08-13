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
    # ... (保持字体和样式设置不变) ...

    # 设置模型简称映射
    model_names = {
        "mobilenetv2_100": "MobileNetV2",
        "vit_base_patch16_siglip_224": "ViT-Base",
        "resnet101": "ResNet101"
    }

    models = df['model'].unique().tolist()
    preprocessors = ['cpu', 'gpu']
    percentiles = [50, 90, 99, 99.9, 99.99, 99.999]
    percentile_labels = ['50', '90', '99', '99.9', '99.99', '99.999']

    # 增强视觉区分度
    line_styles = {'cpu': '-', 'gpu': '--'}
    markers = {'cpu': 'o', 'gpu': 's'}
    colors = {'cpu': '#1f77b4', 'gpu': '#d62728'}

    height_ratios = []
    model_ranges = []

    for model in models:
        model_df = df[df['model'] == model]
        all_vals = model_df[percentile_labels].values
        data_min = np.min(all_vals)
        data_max = np.max(all_vals)
        padding = (data_max - data_min) * 0.05 + 0.5
        y_min = max(0.1, data_min - padding)
        y_max = data_max + padding
        model_ranges.append((y_min, y_max))
        height_ratios.append(y_max - y_min)

    with PdfPages(output_file) as pdf:
        fig = plt.figure(figsize=(8, 10))

        # 关键修改：设置垂直间距为0
        gs = plt.GridSpec(len(models), 1, figure=fig,
                          height_ratios=height_ratios, hspace=0)  # hspace=0去除间隔
        axes = [fig.add_subplot(gs[i]) for i in range(len(models))]

        for i, model in enumerate(models):
            ax = axes[i]
            model_df = df[df['model'] == model]
            model_short = model_names.get(model, model)
            y_min, y_max = model_ranges[i]

            for pp in preprocessors:
                pp_df = model_df[model_df['preprocessor'] == pp]
                if not pp_df.empty:
                    values = pp_df.iloc[0][percentile_labels].values
                    ax.plot(percentile_labels, values,
                            marker=markers[pp],
                            markersize=6,
                            linestyle=line_styles[pp],
                            color=colors[pp],
                            label=f'{pp.upper()} Preprocessor',
                            zorder=3,
                            markeredgewidth=1.2)

            num_clients = model_df.iloc[0]['num_clients']
            ax.text(0.98, 0.92, f"Concurrency: {num_clients}",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontstyle='italic',
                    color='#444444',
                    horizontalalignment='right',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white',
                              alpha=0.85,
                              edgecolor='#bbbbbb',
                              linewidth=0.8))

            ax.grid(True, which="major", linestyle='--', alpha=0.7, zorder=1)
            ax.set_ylim(y_min, y_max)

            # 优化科学计数法显示位置
            if y_max > 1000:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                # 将指数标签移到轴顶部避免重叠
                ax.yaxis.get_offset_text().set_position((0, 1.02))
                ax.yaxis.get_offset_text().set_fontsize(10)

            # 移除子图之间的边框线
            if i < len(models) - 1:
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(bottom=False, labelbottom=False)

            # 只在最下方子图显示x轴标签
            if i == len(models) - 1:
                ax.set_xlabel('Percentile', fontsize=12, labelpad=8)
            else:
                ax.set_xticklabels([])

            # 调整子图标签位置避免重叠
            ax.text(0.02, 0.92, f'({chr(97+i)}) {model_short}',
                    transform=ax.transAxes,
                    fontsize=16,
                    fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white',
                              ec='none',
                              alpha=0.8))

        # 添加共享的y轴标签
        fig.text(0.04, 0.5, 'Latency (ms)', va='center',
                 rotation='vertical', fontsize=12, fontweight='bold')

        # 创建统一的图例
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels,
                            loc='lower center',
                            ncol=2,
                            frameon=True,
                            framealpha=0.95,
                            edgecolor='#333333',
                            facecolor='white',
                            bbox_to_anchor=(0.5, 0.01),
                            title_fontsize=11)
        legend.get_frame().set_linewidth(1.0)

        # 调整布局确保图例有足够空间
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.96], pad=1.5)
        fig.subplots_adjust(bottom=0.12)

        pdf.savefig(fig, dpi=600)
        plt.close()


if __name__ == "__main__":
    log_file = "../stability.log"
    output_pdf = "latency_percentiles.pdf"

    df = parse_log_file(log_file)
    plot_percentile_comparison(df, output_pdf)

    print(f"优化后的图表已保存至: {output_pdf}")
