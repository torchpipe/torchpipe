import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# 调整全局字体大小以适应更小的图形
plt.rcParams.update({
    'font.size': 14,           # 减小基础字体大小
    'axes.titlesize': 16,      # 减小坐标轴标题大小
    'axes.labelsize': 15,      # 减小坐标轴标签大小
    'xtick.labelsize': 13,     # 减小x轴刻度标签大小
    'ytick.labelsize': 13,     # 减小y轴刻度标签大小
    'legend.fontsize': 12,     # 减小图例字体大小
    'figure.titlesize': 18     # 减小图形标题大小
})

# Read file and parse data


def parse_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Extract key information using regular expressions
            id_match = re.search(r'id:([^,]+)', line)
            rate_match = re.search(r'rate:([\d.]+)', line)
            cv_match = re.search(r'cv=([\d.]+)', line)
            slo_match = re.search(r'(slo_scale|slo)=([\d.]+)', line)
            goodput_match = re.search(r'goodput speed=([\d.]+)', line)
            good_req_match = re.search(r'good_req=(\d+)', line)
            total_req_match = re.search(r'total_req=(\d+)', line)

            if all([id_match, rate_match, cv_match, slo_match, goodput_match]):
                entry = {
                    'id': id_match.group(1),
                    'rate': float(rate_match.group(1)),
                    'cv': float(cv_match.group(1)),
                    'slo_scale': float(slo_match.group(2)),
                    'goodput_speed': float(goodput_match.group(1))
                }

                # 添加可选的 good_req 和 total_req 字段
                if good_req_match and total_req_match:
                    entry['good_req'] = int(good_req_match.group(1))
                    entry['total_req'] = int(total_req_match.group(1))

                data.append(entry)
    return data

# Plot comparison charts


def plot_comparison(data):
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 创建2行2列的布局，左侧放两个rate scale图，右侧放CV和SLO图
    fig = plt.figure(figsize=(14, 10))

    # 创建网格布局：左侧两图，右侧两图
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          left=0.1, right=0.95, bottom=0.1, top=0.9,
                          wspace=0.3, hspace=0.4)

    # 创建子图
    ax1 = fig.add_subplot(gs[0, 0])  # 第一个rate scale图放在左上
    ax2 = fig.add_subplot(gs[1, 0])  # 第二个rate scale图放在左下
    ax3 = fig.add_subplot(gs[0, 1])  # CV图放在右上
    ax4 = fig.add_subplot(gs[1, 1])  # SLO图放在右下

    # Extract all different IDs
    ids = list(set(entry['id'] for entry in data))
    ids.sort()
    print(f"Found IDs: {ids}")  # 调试信息

    # Define markers and line styles for better distinction
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    line_styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 使用更鲜明的颜色

    # Create a single legend for all plots
    legend_handles = []

    # 收集所有y值以确保y轴刻度大致相等
    all_y_values = []

    # 获取所有不同的slo_scale值
    slo_scales = sorted(set(entry['slo_scale'] for entry in data))
    print(f"Found SLO scales: {slo_scales}")

    # 选择两个不同的slo_scale值
    if len(slo_scales) >= 2:
        slo1, slo2 = slo_scales[0], slo_scales[1]
    else:
        slo1, slo2 = 7.0, 3.0  # 默认值

    # First plot: Compare rate from 1 to 5 (fixed cv=0.5, slo_scale=slo1)
    rate_data1 = {}
    available_rates1 = set()
    for id_name in ids:
        rate_data1[id_name] = {}
        for rate in range(1, 6):
            # Filter data matching the criteria
            filtered = [d for d in data if d['id'] == id_name and
                        d['rate'] == rate and
                        abs(d['cv'] - 0.5) < 0.1 and
                        abs(d['slo_scale'] - slo1) < 0.1]
            if filtered:
                # Calculate average (if multiple data points)
                avg_goodput = sum(d['goodput_speed']
                                  for d in filtered) / len(filtered)
                rate_data1[id_name][rate] = avg_goodput
                all_y_values.append(avg_goodput)
                available_rates1.add(rate)

    # Plot rate comparison for slo1
    available_rates1 = sorted(list(available_rates1))
    for i, id_name in enumerate(ids):
        marker_idx = i % len(markers)
        line_idx = i % len(line_styles)
        color_idx = i % len(colors)
        x_vals = []
        y_vals = []
        for rate in available_rates1:
            if rate in rate_data1[id_name]:
                x_vals.append(rate)
                y_vals.append(rate_data1[id_name][rate])

        if x_vals:
            line, = ax1.plot(x_vals, y_vals, marker=markers[marker_idx],
                             linestyle=line_styles[line_idx], color=colors[color_idx],
                             markersize=6, linewidth=1.5, markeredgewidth=1.2)
            legend_handles.append(line)

    ax1.set_xlabel(
        f'Rate Scale\n(CV=0.5, SLO={slo1})', fontsize=14, labelpad=6)
    ax1.set_ylabel('Goodput (req/s)', fontsize=14, labelpad=6)
    ax1.set_xticks(available_rates1)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Second plot: Compare rate from 1 to 5 (fixed cv=0.5, slo_scale=slo2)
    rate_data2 = {}
    available_rates2 = set()
    for id_name in ids:
        rate_data2[id_name] = {}
        for rate in range(1, 6):
            # Filter data matching the criteria
            filtered = [d for d in data if d['id'] == id_name and
                        d['rate'] == rate and
                        abs(d['cv'] - 0.5) < 0.1 and
                        abs(d['slo_scale'] - slo2) < 0.1]
            if filtered:
                # Calculate average (if multiple data points)
                avg_goodput = sum(d['goodput_speed']
                                  for d in filtered) / len(filtered)
                rate_data2[id_name][rate] = avg_goodput
                all_y_values.append(avg_goodput)
                available_rates2.add(rate)

    # Plot rate comparison for slo2
    available_rates2 = sorted(list(available_rates2))
    for i, id_name in enumerate(ids):
        marker_idx = i % len(markers)
        line_idx = i % len(line_styles)
        color_idx = i % len(colors)
        x_vals = []
        y_vals = []
        for rate in available_rates2:
            if rate in rate_data2[id_name]:
                x_vals.append(rate)
                y_vals.append(rate_data2[id_name][rate])

        if x_vals:
            line, = ax2.plot(x_vals, y_vals, marker=markers[marker_idx],
                             linestyle=line_styles[line_idx], color=colors[color_idx],
                             markersize=6, linewidth=1.5, markeredgewidth=1.2)

    ax2.set_xlabel(
        f'Rate Scale\n(CV=0.5, SLO={slo2})', fontsize=14, labelpad=6)
    ax2.set_ylabel('Goodput (req/s)', fontsize=14, labelpad=6)
    ax2.set_xticks(available_rates2)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Third plot: Compare cv
    cv_data = {}
    available_cvs = set()
    for id_name in ids:
        cv_data[id_name] = {}
        for cv in range(1, 5):
            # Filter data matching the criteria
            filtered = [d for d in data if d['id'] == id_name and
                        abs(d['cv'] - cv) < 0.01]
            if filtered:
                # Calculate average (if multiple data points)
                avg_goodput = sum(d['goodput_speed']
                                  for d in filtered) / len(filtered)
                cv_data[id_name][cv] = avg_goodput
                all_y_values.append(avg_goodput)
                available_cvs.add(cv)

    # Plot cv comparison
    available_cvs = sorted(list(available_cvs))
    for i, id_name in enumerate(ids):
        marker_idx = i % len(markers)
        line_idx = i % len(line_styles)
        color_idx = i % len(colors)
        x_vals = []
        y_vals = []
        for cv in available_cvs:
            if cv in cv_data[id_name]:
                x_vals.append(cv)
                y_vals.append(cv_data[id_name][cv])
        if x_vals:
            line = ax3.plot(x_vals, y_vals, marker=markers[marker_idx],
                            linestyle=line_styles[line_idx], color=colors[color_idx],
                            markersize=6, linewidth=1.5, markeredgewidth=1.2)[0]

    ax3.set_xlabel('CV Scale\n(Rate=3, SLO=7)', fontsize=14, labelpad=6)
    ax3.set_ylabel('Goodput (req/s)', fontsize=14, labelpad=6)
    ax3.set_xticks(available_cvs)
    ax3.tick_params(axis='both', which='major', labelsize=13)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Fourth plot: Compare slo_scale (fixed rate=3, cv=0.5)
    slo_data = {}
    available_slos = set()
    slo_scales = [1.5, 2.0, 2.5, 3.0]
    for id_name in ids:
        slo_data[id_name] = {}
        for slo in slo_scales:
            # Filter data matching the criteria
            filtered = [d for d in data if d['id'] == id_name and
                        abs(d['slo_scale'] - slo) < 0.01]
            if filtered:
                avg_goodput = sum(d['goodput_speed']
                                  for d in filtered) / len(filtered)
                slo_data[id_name][slo] = avg_goodput
                all_y_values.append(avg_goodput)
                available_slos.add(slo)

    # Plot slo_scale comparison
    available_slos = sorted(list(available_slos))
    for i, id_name in enumerate(ids):
        marker_idx = i % len(markers)
        line_idx = i % len(line_styles)
        color_idx = i % len(colors)
        x_vals = []
        y_vals = []
        for slo in available_slos:
            if slo in slo_data[id_name]:
                x_vals.append(slo)
                y_vals.append(slo_data[id_name][slo])

        if x_vals:
            line = ax4.plot(x_vals, y_vals, marker=markers[marker_idx],
                            linestyle=line_styles[line_idx], color=colors[color_idx],
                            markersize=6, linewidth=1.5, markeredgewidth=1.2)[0]

    ax4.set_xlabel('SLO Scale\n(Rate=3, CV=0.5)', fontsize=14, labelpad=6)
    ax4.set_ylabel('Goodput (req/s)', fontsize=14, labelpad=6)
    ax4.set_xticks(available_slos)
    ax4.tick_params(axis='both', which='major', labelsize=13)
    ax4.grid(True, linestyle='--', alpha=0.7)

    # 设置所有y轴的范围大致相等
    y_min = min(all_y_values) * 0.9  # 留出10%的边距
    y_max = max(all_y_values) * 1.1  # 留出10%的边距

    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    ax3.set_ylim(y_min, y_max)
    ax4.set_ylim(y_min, y_max)

    id_name2name = {
        'omniback_sd_256_candrop0': 'Plain',
        'omniback_sd_256_candrop1': 'Profile-Guided Dropping',
        'neustream_sd_256': 'NeuStream'
    }

    legend_labels = [id_name2name[x] for x in ids]

    if legend_handles:
        # 将图例放在左上角空白处
        fig.legend(legend_handles, legend_labels,
                   loc='upper left', ncol=1,
                   bbox_to_anchor=(0.05, 0.85), fontsize=12,
                   frameon=True, fancybox=True, shadow=True)
    else:
        print("Error: Still no legend handles available.")

    # Save to PDF
    with PdfPages('tmp/stable_diffusion_comparison.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    print(f'saved to tmp/stable_diffusion_comparison.pdf')
    plt.show()

# Main function


def main():
    filename = 'stable_diffusion_serve_result.txt'
    data = parse_file(filename)
    print(f"Parsed {len(data)} data points")
    if not data:
        print("Error: No data parsed from file. Please check the file format.")
        return
    plot_comparison(data)


if __name__ == '__main__':
    main()
