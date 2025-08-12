import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 数据来源于表格
concurrency = np.array([1, 3, 5, 8, 10, 20, 40, 80, 160])
qps = np.array([194, 403, 655, 1236, 1445, 2468, 2759, 2664, 2650])
tp50 = np.array([5.1, 7.4, 7.6, 6.4, 6.6, 7.8, 14.7, 29.8, 59.7])
tp99 = np.array([7.2, 7.6, 7.8, 7.5, 8.5, 10.3, 17.6, 34.1, 65.7])
gpu_util = np.array([34, 28, 30, 45, 63, 96, 100, 100, 100])

# 创建图表
plt.figure(figsize=(14, 10))

# 创建TP99主图（左轴）
ax1 = plt.subplot(2, 1, 1)
ax1.set_title('ResNet101 Latency Analysis (TP99 Highlighted)',
              fontsize=14, pad=20)

# 绘制延迟曲线
tp99_line, = ax1.plot(concurrency, tp99, 'o-', color='#E63946',
                      linewidth=3, markersize=10, label='TP99')
tp50_line, = ax1.plot(concurrency, tp50, 's--', color='#457B9D',
                      linewidth=2, markersize=8, label='TP50')

# 高亮显示关键点
highlight_idx = 5  # 并发20的索引
ax1.plot(concurrency[highlight_idx], tp99[highlight_idx], 'o',
         markersize=14, markeredgecolor='black', markeredgewidth=2,
         markerfacecolor='none', zorder=10)
ax1.annotate(f'Optimal Point: {tp99[highlight_idx]}ms @ {concurrency[highlight_idx]} conc.',
             xy=(concurrency[highlight_idx], tp99[highlight_idx]),
             xytext=(30, 50), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))

# 设置坐标轴
ax1.set_xscale('log')
ax1.set_xticks(concurrency)
ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax1.set_xlabel('Request Concurrency', fontsize=12)
ax1.set_ylabel('Latency (ms)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left', fontsize=12)

# 添加QPS和GPU信息（右轴）
ax2 = ax1.twinx()
qps_line, = ax2.plot(concurrency, qps, 'd-', color='#2A9D8F',
                     linewidth=2, markersize=8, label='QPS')
gpu_fill = ax2.fill_between(concurrency, gpu_util,
                            color='#F4A261', alpha=0.2, label='GPU Util (%)')

# 标记饱和点
ax2.axvline(x=40, color='gray', linestyle=':', linewidth=2)
ax2.annotate('GPU Saturation Point', xy=(40, 2700),
             xytext=(50, 2600), arrowprops=dict(arrowstyle='->'))

ax2.set_ylabel('QPS / GPU Utilization (%)', fontsize=12)
ax2.set_ylim(0, 3000)

# 合并图例
lines = [tp99_line, tp50_line, qps_line, gpu_fill]
ax2.legend(lines, [l.get_label()
           for l in lines], loc='upper right', fontsize=12)

# 添加系统性能分析子图
ax3 = plt.subplot(2, 1, 2)
width = 0.35
x = np.arange(len(concurrency))

# 计算延迟增长倍数
tp99_growth = tp99 / tp99[0]
qps_growth = qps / qps[0]

ax3.bar(x - width/2, tp99_growth, width, color='#E63946', label='TP99 Growth')
ax3.bar(x + width/2, qps_growth, width, color='#2A9D8F', label='QPS Growth')

# 标记关键比值
max_ratio_idx = np.argmax(qps_growth / tp99_growth)
ax3.annotate(f'Best Efficiency\n{concurrency[max_ratio_idx]} conc.',
             xy=(max_ratio_idx, qps_growth[max_ratio_idx]),
             xytext=(max_ratio_idx-3, qps_growth[max_ratio_idx]+1),
             arrowprops=dict(arrowstyle='->'))

ax3.set_title('System Efficiency: QPS vs TP99 Growth (Normalized)',
              fontsize=14, pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(concurrency)
ax3.set_xlabel('Request Concurrency', fontsize=12)
ax3.set_ylabel('Growth Multiple (Normalized)', fontsize=12)
ax3.legend(fontsize=12)
ax3.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('resnet_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
