import os
import shutil
import torch
import torchpipe
import fire
import subprocess
import signal
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import numpy as np

def get_idle_gpu() -> int:
    """选择显存使用最小且利用率最低的GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            idx, mem_free, util = map(float, line.split(', '))
            gpu_stats.append({
                'id': int(idx),
                'mem_free': mem_free,
                'util': util
            })

        # 优先选择显存空闲最多的GPU，其次选择利用率最低的
        gpu_stats.sort(key=lambda x: (-x['mem_free'], x['util']))
        selected_gpu = gpu_stats[0]['id']
        print(
            f"Selected GPU {selected_gpu} - Free memory: {gpu_stats[0]['mem_free']} MB, Utilization: {gpu_stats[0]['util']}%")
        return selected_gpu

    except Exception as e:
        print(f"Error selecting GPU, using default 0: {str(e)}")
        return 0


def save_onnx(model: str, gpu_id: int):
    """在指定GPU上导出ONNX模型"""
    onnx_save_path = f"./{model}.onnx"
    if os.path.exists(onnx_save_path):
        return

    import timm
    assert model in timm.list_models(), f"Model {model} not in timm"

    # 设置GPU设备
    device = f"cuda:{gpu_id}"
    torch_model = timm.create_model(
        model, pretrained=False, exportable=True).eval()  # .to(device)

    torchpipe.utils.model_helper.onnx_export(
        torch_model, onnx_save_path, 224, 224)


def build_engine(model: str, batch_size: int, gpu_id: int):
    """在指定GPU上构建TensorRT引擎"""
    engine_path = f"./{model}_b{batch_size}.trt"
    if os.path.exists(engine_path):
        return engine_path

    build_cmd = [
        "trtexec",
        f"--onnx=./{model}.onnx",
        f"--saveEngine={engine_path}",
        f"--shapes=input:{batch_size}x3x224x224",
        "--fp16",
        f"--device={gpu_id}"  # 指定GPU
    ]
    subprocess.run(build_cmd, check=True)
    return engine_path


def run_benchmark(engine_path: str, batch_size: int, gpu_id: int, from_py: bool = False) -> Tuple[float, float, int]:
    """在指定GPU上运行基准测试"""
    # 准备GPU监控
    log_file = f"./gpu_log_{os.path.basename(engine_path)}.csv"

    # nvidia-smi -i=1 --query-gpu=utilization.gpu,memory.used --format=csv -l 1
    # nvidia-smi pmon   -s um -i 1
    # 启动nvidia-smi监控指定GPU
    with open(log_file, 'w') as f:
        proc_smi = subprocess.Popen(
            ['nvidia-smi',
             f'-i={gpu_id}',
             '--query-gpu=utilization.gpu,memory.used',
             '--format=csv', '-l', '1'],
            stdout=f, stderr=subprocess.PIPE
        )

    # 运行基准测试
    if from_py:
        my_env = os.environ.copy()
        my_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        run_cmd = [
            f"python",
            f'{engine_path}',
            f"--batch_size={batch_size}",
            f"--gpu_id={gpu_id}"  # 指定GPU
        ]
    else:
        my_env = os.environ.copy()
        run_cmd = [
            "trtexec",
            f"--loadEngine={engine_path}",
            f"--shapes=input:{batch_size}x3x224x224",
            "--iterations=100",
            f"--device={gpu_id}"  # 指定GPU
        ]

    try:
        result = subprocess.run(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
            env=my_env
        )
    except subprocess.TimeoutExpired:
        print(f"Benchmark timed out for {engine_path}")
        proc_smi.terminate()
        return 0.0, 0.0, 0

    # 停止监控
    proc_smi.terminate()

    # 解析吞吐量
    throughput = 0.0
    for line in result.stdout.split('\n'):
        if "Throughput:" in line:
            try:
                throughput = float(line.split("Throughput:")[1].split('qps')[0].strip().rstrip())
            except (IndexError, ValueError):
                pass
            break

    # 解析GPU指标
    tp50_sm, max_mem = parse_gpu_log(log_file)

    return throughput, tp50_sm, max_mem


def parse_gpu_log(log_path: str) -> Tuple[float, int]:
    """解析GPU日志文件"""
    try:
        with open(log_path) as f:
            lines = f.readlines()[1:]  # 跳过标题行

        sm_vals = []
        mem_vals = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                sm_val = float(parts[0].split()[0])  # 提取利用率数值
                mem_val = int(parts[1].split()[0])   # 提取显存数值
                if sm_val < 10:
                    continue
                sm_vals.append(sm_val)
                mem_vals.append(mem_val)
            except (ValueError, IndexError):
                continue

        # tp50_sm = sum(sm_vals) / len(sm_vals) if sm_vals else 0.0
        tp50_sm = np.percentile(sm_vals, 50) if sm_vals else 0.0
        max_mem = max(mem_vals) if mem_vals else 0
        return tp50_sm, max_mem

    except Exception as e:
        print(f"Error parsing GPU log: {str(e)}")
        return 0.0, 0


# def plot_results(model: str, results: List[dict]):
#     """生成吞吐量对比图"""
#     df = pd.DataFrame(results)
#     if df.empty:
#         return

#     # 找出归一化batch size的吞吐量参考值
#     t32 = df[df['batch_size'] == 32]['throughput'].values
#     t32 = t32[0] if len(t32) > 0 else 0

#     plt.figure(figsize=(10, 6))
#     plt.plot(df['batch_size'], df['throughput'], 'o-', label='Throughput')

#     if t32 > 0:
#         # 绘制75%阈值线并找出最小batch size
#         threshold = 0.75 * t32
#         plt.axhline(threshold, color='r', linestyle='--', label='75% of T32')

#         valid_bs = df[(df['batch_size'] >= 1) &
#                       (df['batch_size'] <= 12) &
#                       (df['throughput'] >= threshold)]
#         if not valid_bs.empty:
#             min_bs = valid_bs['batch_size'].min()
#             plt.plot(min_bs, valid_bs[valid_bs['batch_size'] == min_bs]['throughput'].values[0],
#                      'ro', markersize=8, label=f'Min BS: {min_bs}')

#     plt.title(f'Model: {model}')
#     plt.xlabel('Batch Size')
#     plt.ylabel('Throughput (qps)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{model}_throughput.png")
#     plt.close()

def main(
    models: List[str] = ['seg.py','jpg_decode.py', 'mobilenetv2_100', 'resnet101',
                         'vit_base_patch16_siglip_224'],
    batch_range: Tuple[int, int] = (1, 16),
    norm_batch_size: int = 64
):
    # models: List[str] = ['resnet101', 'mobilenetv2_100',
    #                      'vit_base_patch16_siglip_224'],
    # 验证trtexec是否可用
    if isinstance(models, str):
        models = [models]
    print(f'models = {models}')
    assert shutil.which("trtexec") is not None, "trtexec not found in PATH"

    # 选择最空闲的GPU
    selected_gpu = get_idle_gpu()
    print(f"Using GPU: {selected_gpu} for all benchmarks")

    all_results = []

    for model in models:
        model_results = []
        try:
            if model.endswith('.py'):                
                tput, tp50_sm, max_mem = run_benchmark(
                    model, norm_batch_size, selected_gpu, from_py=True)
                tput *= norm_batch_size
                curr_result = {
                    'model': model,
                    'batch_size': norm_batch_size,
                    'throughput': int(tput),
                    'tp50_sm': tp50_sm,
                    'max_mem': max_mem
                }
                model_results.append(curr_result)
                t32 = tput

                # 基准测试batch size范围
                for bs in range(batch_range[0], batch_range[1] + 1):
                    tput, tp50_sm, max_mem = run_benchmark(
                        model, bs, selected_gpu, from_py=True)
                    tput *= bs
                    current = {
                        'model': model,
                        'batch_size': bs,
                        'throughput': int(tput),
                        'tp50_sm': tp50_sm,
                        'max_mem': max_mem
                    }
                    model_results.append(current)
                    print(f'\ncurrent: {current}\n')
            else:
                save_onnx(model, selected_gpu)

                # 基准测试归一化batch size
                engine_path = build_engine(model, norm_batch_size, selected_gpu)
                tput, tp50_sm, max_mem = run_benchmark(
                    engine_path, norm_batch_size, selected_gpu)
                tput *= norm_batch_size
                curr_result = {
                    'model': model,
                    'batch_size': norm_batch_size,
                    'throughput': int(tput),
                    'tp50_sm': tp50_sm,
                    'max_mem': max_mem
                }
                model_results.append(curr_result)
                t32 = tput

                # 基准测试batch size范围
                for bs in range(batch_range[0], batch_range[1] + 1):
                    engine_path = build_engine(model, bs, selected_gpu)
                    tput, tp50_sm, max_mem = run_benchmark(
                        engine_path, bs, selected_gpu)
                    tput *= bs
                    current = {
                        'model': model,
                        'batch_size': bs,
                        'throughput': int(tput),
                        'tp50_sm': tp50_sm,
                        'max_mem': max_mem
                    }
                    model_results.append(current)
                    print(f'\ncurrent: {current}\n')

            all_results.extend(model_results)
                
        except Exception as e:
            print(f"Error processing {model}: {str(e)}")
        
    # 保存综合结果
    csv_file = "benchmark_results.csv"
    file_exists = os.path.isfile(csv_file)
    pd.DataFrame(all_results).to_csv(
        csv_file, 
        mode='a',          # 追加模式
        index=False,       # 不写入行索引
        header=not file_exists  # 如果文件不存在才写入列名
    )
    print(f'result saved to {csv_file}')


if __name__ == "__main__":
    fire.Fire(main)

