from modules import ClipModule, UNetModule, VaeModule, SafetyModule
import modules
import torch
import time
import json
import os
import fire
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
# from stable_diffusion_v1_5.stable_diffusion_pipeline import StableDiffusionPipeline
from typing import List, Dict, Any, Tuple
import numpy as np
import sys

torch.set_grad_enabled(False)

sys.path.insert(0, './')
assert os.environ["USE_TRT"]


class StableDiffusionPipeline():
    def __init__(self, config_path, **kwargs):
        super().__init__()
        fp = open(config_path, "r")
        config = json.load(fp)
        self.stream_module_list = []
        print(config.keys())
        torch.set_grad_enabled(False)

        self.modules = {'ClipModule': ClipModule,
                        'UNetModule': UNetModule,
                        'VaeModule': VaeModule,
                        'SafetyModule': SafetyModule}

        for key, value in config.items():
            self.modules[key] = self.modules[key](**value)
            # self.modules[key].deploy()
        self.stream_module_list = [self.modules['ClipModule'], self.modules['UNetModule'],
                                   self.modules['VaeModule'], self.modules['SafetyModule']]
        self.default_deploy()

    def default_deploy(self, **kwargs):
        for module in self.stream_module_list:
            module.deploy()


# 初始化管道
sd_config_file = "stable_diffusion_v1_5/config.json"
sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)

sd_modules = sd_pipeline.stream_module_list


def warmup_modules(
    modules: List,
    request_template: Dict[str, Any],
    warmup_iters: int = 5,
    progress_bar: bool = True
):
    """预热模块以稳定性能"""
    if progress_bar:
        print("\n🔥 Warming up modules...")
        pbar = tqdm(total=warmup_iters * len([1, 4, 8]) * len(modules),
                    desc="Warmup Progress", unit="op")

    for _ in range(warmup_iters):
        for batch_size in [1, 4, 8, 1]:  # 使用不同批次大小进行预热
            requests = [deepcopy(request_template) for _ in range(batch_size)]
            for module in modules:
                module.compute(requests)
                for req in requests:
                    for k, v in req.items():
                        if type(v) == torch.Tensor:
                            assert v.is_cpu == False
                torch.cuda.synchronize()
                if progress_bar:
                    pbar.update(1)

    if progress_bar:
        pbar.close()
        print("✅ Warmup completed")


def save_partial_results(results: Dict, output_path: str):
    """保存部分结果到文件"""
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=True)


def profile_module_performance(
    modules: List,
    batch_sizes: List[int],
    request_template: Dict[str, Any],
    num_trials: int = 5,
    progress_bar: bool = True,
    output_path: str = None
) -> Dict[str, Dict[str, float]]:
    """性能测试函数，多次测量求平均，并增量保存结果"""
    latency_profiles = {type(m).__name__: {} for m in modules}

    # 创建进度条
    if progress_bar:
        total_ops = len(batch_sizes) * num_trials * len(modules)
        pbar = tqdm(total=total_ops, desc="Profiling Progress",
                    unit="batch-module")

    import sys
    sys.path.insert(
        0, './25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/H100_SD_FP16_img512/')
    from test_set import prompt_list
    index = 0

    for batch_size in batch_sizes:
        # 存储每次试验的结果
        trial_results = {type(m).__name__: [] for m in modules}

        for trial in range(num_trials):
            # 创建请求批次（深拷贝避免引用问题）
            requests = []
            for i in range(batch_size):
                req = deepcopy(request_template)
                req['prompt'] = prompt_list[index % 100]
                index += 1
                requests.append(req)

            for module in modules:
                # 确保CUDA操作同步
                if type(module).__name__ == "UNetModule":
                    unet_loop_num = request_template["loop_num"].get(
                        "UNetModule", 50)
                else:
                    unet_loop_num = 1
                torch.cuda.current_stream().synchronize()

                # 精确测量执行时间
                start_time = time.perf_counter()
                for _ in range(unet_loop_num):
                    module.compute(requests)
                torch.cuda.current_stream().synchronize()

                # 计算耗时（毫秒）
                elapsed = (time.perf_counter() - start_time)

                # 如果是UNet模块，除以循环次数
                elapsed /= unet_loop_num

                trial_results[type(module).__name__].append(elapsed)

                # 更新进度条
                if progress_bar:
                    pbar.update(1)

        # 计算平均延迟
        for module_name, latencies in trial_results.items():
            if len(latencies) < 10:
                latencies += latencies
            avg_latency = np.mean(np.sort(latencies)[int(
                len(latencies)*0.1): -int(len(latencies)*0.1)])
            latency_profiles[module_name][str(batch_size)] = float(
                f"{avg_latency:.8f}")

        # 增量保存当前结果
        if output_path:
            new_latency_profiles = {}
            for module_name, v in latency_profiles.items():
                new_latency_profiles[module_name.replace(
                    'Module', '').lower()] = v
            save_partial_results(new_latency_profiles, output_path)

    new_latency_profiles = {}
    for module_name, v in latency_profiles.items():
        new_latency_profiles[module_name.replace('Module', '').lower()] = v

    if progress_bar:
        pbar.close()

    return new_latency_profiles


def generate_filename(image_size: int, config: Dict, max_batch: int) -> str:
    """生成包含配置信息的文件名"""
    import json
    import toml
    # with open('data/test_config.toml', 'r') as f:
    #     data = toml.load(f)
    # EXP_ID = os.getenv('EXP_ID')
    # json_p = data[EXP_ID]['latency_profile']

    return 'profiles/latency_profile.json'


def main(
    image_size: int = 256,
    min_batch: int = 1,
    max_batch: int = 48,
    num_trials: int = 5,
    warmup_iters: int = 3,
    config_path: str = "stable_diffusion_v1_5/config.json",
    output_dir: str = "profiles",
    progress_bar: bool = True
):
    """
    Stable Diffusion 模块性能分析工具
    
    参数:
    image_size (int): 图像尺寸 (默认: 256)
    min_batch (int): 最小批处理大小 (默认: 1)
    max_batch (int): 最大批处理大小 (默认: 40)
    num_trials (int): 每个批处理大小的测试次数 (默认: 5)
    warmup_iters (int): 预热迭代次数 (默认: 3)
    config_path (str): 配置文件路径 (默认: "stable_diffusion_v1_5/config.json")
    output_dir (str): 输出目录 (默认: "profiles")
    progress_bar (bool): 是否显示进度条 (默认: True)
    """
    print(f"\n🚀 Starting performance profiling with configuration:")
    print(f"  Image Size: {image_size}x{image_size}")
    print(f"  Batch Sizes: {min_batch} to {max_batch}")
    print(f"  Trials per Batch: {num_trials}")
    print(f"  Warmup Iterations: {warmup_iters}")
    print(f"  Config Path: {config_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 配置和初始化
    # pipeline = initialize_pipeline(config_path)
    modules = sd_modules

    prompt = "a boy studying in Chinese University"
    # 请求模板
    REQUEST_TEMPLATE = {
        "prompt": prompt,
        "height": image_size,
        "width": image_size,
        "loop_num": {"UNetModule": 30},  # UNet循环50次
        "guidance_scale": 7.5,
        "seed": 81,
        "SLO": 10000,
        "loop_index": {"UNetModule": 0},
        "id": 1,
        "request_time": time.time()
    }

    # 性能测试参数
    BATCH_SIZES = list(range(min_batch, max_batch + 1))

    # 生成输出文件名
    filename = generate_filename(image_size, REQUEST_TEMPLATE, max_batch)
    output_path = filename  # os.path.join(output_dir, filename)

    # 预热模块
    warmup_modules(modules, REQUEST_TEMPLATE, warmup_iters, progress_bar)

    # 执行性能测试，并增量保存结果
    if progress_bar:
        print("\n📊 Starting performance profiling...")
    latency_results = profile_module_performance(
        modules=modules,
        batch_sizes=BATCH_SIZES,
        request_template=REQUEST_TEMPLATE,
        num_trials=num_trials,
        progress_bar=progress_bar,
        output_path=output_path
    )

    # 输出结果摘要
    print(f"\n✅ Profiling completed! Results saved to: {output_path}")
    print("\n📋 Summary of average latencies (s):")
    for module, data in latency_results.items():
        min_batch_latency = data.get(str(min_batch), "N/A")
        max_batch_latency = data.get(str(max_batch), "N/A")
        print(f"  {module}:")
        print(f"    Batch {min_batch}: {min_batch_latency}")
        print(f"    Batch {max_batch}: {max_batch_latency}")

    return output_path


if __name__ == "__main__":
    fire.Fire(main)
