import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv('benchmark_results.csv')

# 获取batch_size=32时的基准吞吐量
base_throughput = df[df['batch_size'] == 32]['throughput'].values[0]

# 计算归一化吞吐量
df['normalized_throughput'] = df['throughput'] / base_throughput

# 提取batch_size小于32的数据点（排除基准点）
plot_df = df[df['batch_size'] < 32].sort_values('batch_size')

# 创建图形
plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)  # 基准线

# 绘制归一化吞吐量曲线
plt.plot(plot_df['batch_size'], plot_df['normalized_throughput'], 
         'o-', color='b', linewidth=2, markersize=8, 
         label='Normalized Throughput')

# 添加75%性能阈值线
plt.axhline(y=0.75, color='r', linestyle='--', label='75% of BS32 Throughput')

# 找出第一个达到75%性能的batch size
valid_points = plot_df[plot_df['normalized_throughput'] >= 0.75]
if not valid_points.empty:
    min_bs = valid_points['batch_size'].min()
    min_point = valid_points[valid_points['batch_size'] == min_bs].iloc[0]
    plt.plot(min_bs, min_point['normalized_throughput'], 
             'ro', markersize=10, label=f'Min BS: {min_bs}')
    
    # 添加文本标注
    plt.text(min_bs, min_point['normalized_throughput'] + 0.03, 
             f'BS={min_bs}', ha='center', fontsize=12)

# 设置图表属性
plt.title('Normalized Throughput vs Batch Size', fontsize=14)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Normalized Throughput (BS=32=1.0)', fontsize=12)
plt.xticks(np.arange(1, 17, 1))
plt.ylim(0, 1.1)
plt.legend(loc='lower right', fontsize=10)

# 添加数据标签
for i, row in plot_df.iterrows():
    plt.text(row['batch_size'], row['normalized_throughput'] + 0.02, 
             f"{row['normalized_throughput']:.2f}", 
             ha='center', fontsize=9)

# 保存图表
plt.tight_layout()
plt.savefig('normalized_throughput.png', dpi=300)
plt.show()
