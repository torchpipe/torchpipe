
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# 设置全局样式
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def parse_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # 使用更精确的正则表达式
            id_match = re.search(r'id:([^,\s]+)', line)
            rate_match = re.search(r'rate:([\d.]+)', line)
            cv_match = re.search(r'cv=([\d.]+)', line)
            slo_match = re.search(r'(slo_scale|slo)=([\d.]+)', line)
            goodput_match = re.search(r'goodput speed=([\d.]+)', line)
            good_req_match = re.search(r'good_req=(\d+)', line)
            total_req_match = re.search(r'total_req=(\d+)', line)

            try:
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
            except ValueError as e:
                print(f"Error parsing line: {line.strip()}")
                print(f"Error: {e}")
    return data


def plot_comparison(data):
    # 创建2行2列的布局
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axs.flat

    # 调整子图间距
    plt.subplots_adjust(wspace=0.25, hspace=0.3, left=0.1,
                        right=0.95, bottom=0.15, top=0.9)

    # 提取所有不同的ID
    ids = list(set(entry['id'] for entry in data))
    ids.sort()
    print(f"Found IDs: {ids}")

    # 定义标记和线条样式
    markers = ['o', 's', 'D', '^']
    line_styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 创建图例句柄
    legend_handles = []
    all_y_values = []

    # 左上图: Compare rate from 1 to 5 (fixed cv=0.5, slo_scale=7.0)
    rate_data_7 = {}
    available_rates_7 = set()
    for id_name in ids:
        rate_data_7[id_name] = {}
        for rate in range(1, 6):
            filtered = [d for d in data if d['id'] == id_name and
                        d['rate'] == rate and
                        abs(d['cv'] - 0.5) < 0.1 and
                        abs(d['slo_scale'] - 7.0) < 0.1]
            if filtered:
                avg_goodput = sum(d['goodput_speed']
                                  for d in filtered) / len(filtered)
                rate_data_7[id_name][rate] = avg_goodput
                all_y_values.append(avg_goodput)
                available_rates_7.add(rate)

    available_rates_7 = sorted(list(available_rates_7))
    for i, id_name in enumerate(ids):
        marker_idx = i % len(markers)
        line_idx = i % len(line_styles)
        color_idx = i % len(colors)
        x_vals = []
        y_vals = []
        for rate in available_rates_7:
            if rate in rate_data_7[id_name]:
                x_vals.append(rate)
                y_vals.append(rate_data_7[id_name][rate])

        if x_vals:
            line, = ax1.plot(x_vals, y_vals, marker=markers[marker_idx],
                             linestyle=line_styles[line_idx], color=colors[color_idx],
                             markersize=6, linewidth=1.5, markeredgewidth=1.2)
            legend_handles.append(line)

    ax1.set_xlabel('Rate Scale\n(CV=0.5, SLO=7)', fontsize=12, labelpad=6)
    ax1.set_ylabel('Goodput (req/s)', fontsize=12, labelpad=6)
    ax1.set_xticks(available_rates_7)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 右上图: Compare cv
    cv_data = {}
    available_cvs = set()
    for id_name in ids:
        cv_data[id_name] = {}
        for cv in [0.5, 1.0, 2.0, 3.0, 4.0]:  # 使用浮点数而不是整数
            filtered = [d for d in data if d['id'] == id_name and
                        abs(d['cv'] - cv) < 0.1 and
                        abs(d['slo_scale'] - 7.0) < 0.1 and
                        abs(d['rate'] - 3.0) < 0.1]
            if filtered:
                avg_goodput = sum(d['goodput_speed']
                                  for d in filtered) / len(filtered)
                cv_data[id_name][cv] = avg_goodput
                all_y_values.append(avg_goodput)
                available_cvs.add(cv)

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
            ax2.plot(x_vals, y_vals, marker=markers[marker_idx],
                     linestyle=line_styles[line_idx], color=colors[color_idx],
                     markersize=6, linewidth=1.5, markeredgewidth=1.2)

    ax2.set_xlabel('CV Scale\n(Rate=3, SLO=7)', fontsize=12, labelpad=6)
    ax2.set_xticks(available_cvs)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 左下图: Compare rate from 1 to 5 (fixed cv=0.5, slo_scale=3.0)
    rate_data_3 = {}
    available_rates_3 = set()
    for id_name in ids:
        rate_data_3[id_name] = {}
        for rate in range(1, 6):
            filtered = [d for d in data if d['id'] == id_name and
                        d['rate'] == rate and
                        abs(d['cv'] - 0.5) < 0.1 and
                        abs(d['slo_scale'] - 3.0) < 0.1]
            if filtered:
                avg_goodput = sum(d['goodput_speed']
                                  for d in filtered) / len(filtered)
                rate_data_3[id_name][rate] = avg_goodput
                all_y_values.append(avg_goodput)
                available_rates_3.add(rate)

    available_rates_3 = sorted(list(available_rates_3))
    for i, id_name in enumerate(ids):
        marker_idx = i % len(markers)
        line_idx = i % len(line_styles)
        color_idx = i % len(colors)
        x_vals = []
        y_vals = []
        for rate in available_rates_3:
            if rate in rate_data_3[id_name]:
                x_vals.append(rate)
                y_vals.append(rate_data_3[id_name][rate])

        if x_vals:
            ax3.plot(x_vals, y_vals, marker=markers[marker_idx],
                     linestyle=line_styles[line_idx], color=colors[color_idx],
                     markersize=6, linewidth=1.5, markeredgewidth=1.2)

    ax3.set_xlabel('Rate Scale\n(CV=0.5, SLO=3)', fontsize=12, labelpad=6)
    ax3.set_ylabel('Goodput (req/s)', fontsize=12, labelpad=6)
    ax3.set_xticks(available_rates_3)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 右下图: Compare slo_scale (fixed rate=3, cv=0.5)
    slo_data = {}
    available_slos = set()
    slo_scales = [1.5, 2.0, 2.5, 3.0]
    for id_name in ids:
        slo_data[id_name] = {}
        for slo in slo_scales:
            filtered = [d for d in data if d['id'] == id_name and
                        abs(d['cv'] - 0.5) < 0.1 and
                        abs(d['rate'] - 3) < 0.1 and
                        abs(d['slo_scale'] - slo) < 0.1]
            if filtered:
                avg_goodput = sum(d['goodput_speed']
                                  for d in filtered) / len(filtered)
                slo_data[id_name][slo] = avg_goodput
                all_y_values.append(avg_goodput)
                available_slos.add(slo)

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
            ax4.plot(x_vals, y_vals, marker=markers[marker_idx],
                     linestyle=line_styles[line_idx], color=colors[color_idx],
                     markersize=6, linewidth=1.5, markeredgewidth=1.2)

    ax4.set_xlabel('SLO Scale\n(Rate=3, CV=0.5)', fontsize=12, labelpad=6)
    ax4.set_xticks(available_slos)
    ax4.grid(True, linestyle='--', alpha=0.7)

    # 设置所有y轴的范围
    if all_y_values:
        y_min = min(all_y_values) * 0.9
        y_max = max(all_y_values) * 1.1
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_ylim(y_min, y_max)

    # 设置图例
    id_name2name = {
        'omniback_sd_256_candrop0': 'Plain',
        'omniback_sd_256_candrop1': 'Plain + Profile-Guided Dropping',
        'neustream_sd_256': 'NeuStream'
    }

    legend_labels = [id_name2name.get(x, x) for x in ids]

    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='lower center', ncol=min(3, len(ids)),
                   bbox_to_anchor=(0.5, 0.02), fontsize=11,
                   frameon=True, fancybox=True, shadow=True)
    else:
        print("Warning: No legend handles available.")

    # 保存到PDF
    with PdfPages('tmp/stable_diffusion_comparison.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    print('Saved to tmp/stable_diffusion_comparison.pdf')
    plt.show()


def main():
    filename = 'stable_diffusion_serve_result_back.txt'
    data = parse_file(filename)
    print(f"Parsed {len(data)} data points")
    if not data:
        print("Error: No data parsed from file. Please check the file format.")
        return
    plot_comparison(data)


if __name__ == '__main__':
    main()
