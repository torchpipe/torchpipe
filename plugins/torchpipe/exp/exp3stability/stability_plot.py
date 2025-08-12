import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib import rcParams
import re

# 设置学术论文风格
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300
rcParams['figure.figsize'] = (6.5, 4.0)  # 增加高度以容纳更多内容
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = ':'
rcParams['grid.alpha'] = 0.3
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05

# 定义MLSys友好的配色方案
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def parse_log(file_path):
    """改进的日志解析函数，正确处理特殊字符"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # 使用正则表达式分割键值对
            parts = re.split(r',(?=\w)', line.strip())
            record = {}
            for part in parts:
                # 处理包含双冒号的情况
                if '::' in part:
                    key, value = part.split('::', 1)
                    # 分离指标和数值
                    if ':' in value:
                        subkey, val = value.split(':', 1)
                        key = f"{key}_{subkey}"
                        value = val
                elif ':' in part:
                    key, value = part.split(':', 1)

                # 规范化键名
                key = key.replace('-', 'neg').replace('.', '_').strip()
                record[key] = value.strip()

            # 添加模型实例标识
            model = record['model']
            instance_id = sum(1 for d in data if d.get('model') == model) + 1
            record['instance_id'] = instance_id
            data.append(record)

    return pd.DataFrame(data)


def plot_stability(df):
    """绘制延迟分布图表 - 横坐标为延迟，纵坐标为百分位指标"""
    fig, ax = plt.subplots()

    # 定义延迟指标和对应的显示名称（纵坐标）
    latency_metrics = [
        'latency_TP99_99', 'latency_TP99_9', 'latency_TP99',
        'latency_TP90', 'latency_TP50'
    ]
    display_names = ['TP99.99', 'TP99.9', 'TP99', 'TP90', 'TP50']

    # 获取所有模型
    models = df['model'].unique()

    # 为每个模型创建数据集
    model_data = {}
    for model in models:
        model_df = df[df['model'] == model]
        model_data[model] = {}
        for metric in latency_metrics:
            # 提取指标值并排序
            values = model_df[metric].astype(float).sort_values().values
            model_data[model][metric] = values

    # 创建纵坐标位置
    y_pos = np.arange(len(latency_metrics))

    # 绘制每个模型的延迟分布
    for i, model in enumerate(models):
        # 计算每个百分位点的中位数延迟
        medians = [np.median(model_data[model][metric])
                   for metric in latency_metrics]

        # 绘制中位线
        ax.plot(
            medians, y_pos,
            color=COLORS[i],
            marker='o',
            markersize=6,
            linewidth=1.5,
            alpha=0.8,
            label=model
        )

        # 添加箱线图显示分布
        for j, metric in enumerate(latency_metrics):
            values = model_data[model][metric]
            # 计算箱线图统计量
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            whisker_low = np.max([np.min(values), q1 - 1.5*iqr])
            whisker_high = np.min([np.max(values), q3 + 1.5*iqr])

            # 绘制箱线图元素
            # 主体箱体
            ax.fill_betweenx(
                [y_pos[j]-0.15, y_pos[j]+0.15],
                q1, q3,
                color=COLORS[i], alpha=0.2
            )

            # 中位线
            ax.plot(
                [medians[j], medians[j]],
                [y_pos[j]-0.15, y_pos[j]+0.15],
                color=COLORS[i], linewidth=1.5
            )

            # 须线
            ax.plot(
                [whisker_low, q1],
                [y_pos[j], y_pos[j]],
                color=COLORS[i], linewidth=1.0, linestyle='--'
            )
            ax.plot(
                [q3, whisker_high],
                [y_pos[j], y_pos[j]],
                color=COLORS[i], linewidth=1.0, linestyle='--'
            )

            # 端点
            ax.scatter(
                whisker_low, y_pos[j],
                s=20, color=COLORS[i], marker='|'
            )
            ax.scatter(
                whisker_high, y_pos[j],
                s=20, color=COLORS[i], marker='|'
            )

    # 设置图表格式
    ax.set_xlabel('Latency (ms)', fontsize=10)
    ax.set_ylabel('Percentile Metrics', fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)

    # 设置x轴范围和对数刻度（因为延迟范围可能很大）
    all_values = np.concatenate(
        [df[metric].astype(float).values for metric in latency_metrics])
    x_min = max(0.1, np.min(all_values) * 0.8)
    x_max = np.max(all_values) * 1.2
    ax.set_xlim(x_min, x_max)
    ax.set_xscale('log')

    # 添加网格和边框
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # 添加图例
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(models),
        frameon=True,
        framealpha=0.9
    )

    # 添加标题
    plt.title('Latency Distribution by Percentile', fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig('latency_distribution.pdf', dpi=300)
    plt.close()


if __name__ == "__main__":
    # 从日志文件读取数据
    df = parse_log("../stability.log")

    # 转换数值列
    numeric_cols = [col for col in df.columns if col not in [
        'model', 'tool_s version']]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 生成图表
    plot_stability(df)
