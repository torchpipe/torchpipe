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
rcParams['figure.figsize'] = (6.5, 4.0)
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
    """绘制延迟分布图表 - 横坐标为百分位指标，纵坐标为延迟"""
    fig, ax = plt.subplots()
    
    # 定义百分位点顺序
    percentiles = ['TP50', 'TP90', 'TP99', 'TP99.9', 'TP99.99']
    metric_keys = ['latency_TP50', 'latency_TP90', 'latency_TP99', 
                   'latency_TP99_9', 'latency_TP99_99']
    
    # 获取所有模型
    models = df['model'].unique()
    
    # 设置横坐标位置
    x_pos = np.arange(len(percentiles))
    bar_width = 0.15  # 每个模型的柱宽
    offset = np.linspace(-0.3, 0.3, len(models))  # 每个模型的偏移量
    
    # 绘制每个模型的箱线图
    for i, model in enumerate(models):
        model_data = []
        for metric in metric_keys:
            # 获取当前模型当前指标的所有值
            values = df[df['model'] == model][metric].astype(float)
            model_data.append(values)
        
        # 绘制箱线图（位置偏移以避免重叠）
        positions = x_pos + offset[i]
        box = ax.boxplot(
            model_data, 
            positions=positions,
            widths=bar_width,
            patch_artist=True,
            showfliers=False,  # 不显示异常值
            boxprops=dict(facecolor=COLORS[i], alpha=0.7, linewidth=0.7),
            medianprops=dict(color='white', linewidth=1.2),
            whiskerprops=dict(color=COLORS[i], linewidth=1.0),
            capprops=dict(color=COLORS[i], linewidth=1.0)
        
        # 添加中位线连接
        medians = [np.median(vals) for vals in model_data]
        ax.plot(
            positions, medians, 
            color=COLORS[i], 
            linestyle='-', 
            marker='o', 
            markersize=4,
            linewidth=1.5,
            alpha=0.9,
            label=model
        )
    
    # 设置图表格式
    ax.set_xlabel('Percentile Metrics', fontsize=10)
    ax.set_ylabel('Latency (ms)', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(percentiles)
    
    # 设置纵坐标范围和对数刻度
    all_values = np.concatenate([df[metric].astype(float).values for metric in metric_keys])
    y_min = max(0.1, np.min(all_values) * 0.8)
    y_max = np.max(all_values) * 1.2
    ax.set_ylim(y_min, y_max)
    ax.set_yscale('log')
    
    # 添加次要刻度线
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(1.0, 10.0) * 0.1))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    
    # 添加网格和边框
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    
    # 添加图例
    ax.legend(
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        edgecolor='#DDDDDD'
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
    numeric_cols = [col for col in df.columns if col not in ['model', 'tool_s version']]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # 生成图表
    plot_stability(df)