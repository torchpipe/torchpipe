import json
import numpy as np


class GammaProcess:
    """Gamma arrival process."""

    def __init__(self, arrival_rate: float, cv: float):
        """Initialize a gamma arrival process.

        Args:
            arrival_rate: mean arrival rate.
            cv: coefficient of variation. When cv == 1, the arrival process is
                Poisson process.
        """
        self.rate_ = arrival_rate  # time interval = mean, so rate = 1 / mean
        self.cv_ = cv  # cv = sigma / mean = 1 / sqrt(k)
        self.shape = 1 / (cv * cv)  # shape = k
        self.scale = cv * cv / arrival_rate  # scale = theta

    def rate(self):
        return self.rate_

    def cv(self):
        return self.cv_

    def generate_arrival_intervals_yhc(self, arrival_rate: float, cv: float, request_count: int, seed: int = 0):
        np.random.seed(seed)
        shape = 1 / (cv * cv)
        scale = cv * cv / arrival_rate
        intervals = np.round(np.random.gamma(
            shape, scale, size=request_count), 3).tolist()
        return intervals


seed = 0
request_count = 500

# 生成随机步长列表
np.random.seed(seed)
random_step_list = np.random.randint(30, 51, request_count)

trace = {
    "random_step_list": random_step_list.tolist(),
}

process = GammaProcess(1.0, 1.0)

# 定义到达率和变异系数列表
arrival_rate_list = [1, 1.5, 1.75, 2.0, 2.25, 2.5]  # req/s
arrival_rate_list += [2.5 + 0.25 * i for i in range(1, 50)]
cv_list = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]  # 突发程度指标

# 生成各参数组合的到达间隔
for arrival_rate in arrival_rate_list:
    for cv in cv_list:
        trace_key = f"rate={arrival_rate:.2f},cv={cv:.2f}"
        # print(f'trace_key={trace_key}')
        trace[trace_key] = process.generate_arrival_intervals_yhc(
            arrival_rate, cv, request_count, seed
        )
import torch
gpu_name = torch.cuda.get_device_name(0).replace(
    'NVIDIA', '').replace(' ', '').replace('GeForceRTX', '')
name = f"data/{gpu_name}_SD_FP16_img256_trace.json"
name = f"data/SD_FP16_img256_trace.json"
f = open(name, "w")

json.dump(trace, f)
f.close()  # 确保关闭文件
print(f'saved to {name}')
