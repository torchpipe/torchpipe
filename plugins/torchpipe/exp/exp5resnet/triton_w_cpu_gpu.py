import os
import tempfile
import pickle
import subprocess
import argparse
from tqdm import tqdm
import time
import signal

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cmd",
    dest="cmd",
    type=str,
    default="",
    help="basic cmd",
)
parser.add_argument(
    "--num_clients",
    dest="num_clients",
    type=str,
    default="1, 5, 10, 20, 30",
    help="basic num_clients",
)
parser.add_argument(
    "--gpu_id",
    dest="gpu_id",
    type=str,
    default="0",
    help="GPU ID to use for Triton server",
)
args = parser.parse_args()

num_clients = [int(x.strip()) for x in args.num_clients.split(",")]


def parse_result(result):
    return result.strip().split("-------------------------------------------------------------------")


def start_triton_server(model_repo_cmd, gpu_id):
    """启动Triton服务器并返回进程对象"""
    # 设置CUDA_VISIBLE_DEVICES环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_id

    print(
        f"Starting Triton server with command: {model_repo_cmd} on GPU {gpu_id}")

    # 拆分命令字符串为列表
    cmd_parts = model_repo_cmd.strip().split()
    process = subprocess.Popen(
        cmd_parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # 创建新的进程组，便于后续终止整个进程树
        env=env  # 传递环境变量
    )

    # 等待服务器启动
    time.sleep(10)  # 等待10秒让服务器启动
    print("Triton server started (waiting 10 seconds for initialization)")
    return process


def stop_triton_server(process):
    """停止Triton服务器进程"""
    if process:
        try:
            # 终止整个进程组
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
            print("Triton server stopped successfully")
        except subprocess.TimeoutExpired:
            print("Triton server did not stop gracefully, forcing termination")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            print("Triton server process already terminated")
        except Exception as e:
            print(f"Error stopping Triton server: {e}")


def run_gpu_preprocess_cmd():
    files = []
    print("Running GPU preprocess commands...")
    for i in tqdm(num_clients, desc="GPU Preprocess"):
        total_number = 5000 if i == 1 else 20000
        cmd = [
            "python3",
            "./benchmark.py",
            "--model", "resnet101",
            "--preprocess", "gpu",
            "--preprocess-instances", "8",
            "--max", "5",
            "--trt_instance_num", "2",
            "--timeout", "2",
            "--total_number", str(total_number),
            "--client", str(i),
        ]

        print(f"\nRunning command for {i} clients:")
        print(" ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print(f"Result for {i} clients:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)

        parsed_result = parse_result(result.stdout)
        files.append(parsed_result)

    return files


def run_cpu_preprocess_cmd():
    files = []
    print("Running CPU preprocess commands...")
    for i in tqdm(num_clients, desc="CPU Preprocess"):
        total_number = 5000 if i == 1 else 20000
        cmd = [
            "python3",
            "./benchmark.py",
            "--model", "resnet101",
            "--preprocess-instances", "8",
            "--max", "5",
            "--trt_instance_num", "2",
            "--timeout", "2",
            "--total_number", str(total_number),
            "--client", str(i),
        ]

        print(f"\nRunning command for {i} clients:")
        print(" ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print(f"Result for {i} clients:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)

        parsed_result = parse_result(result.stdout)
        files.append(parsed_result)
        print(f"Parsed result: {parsed_result}")

    return files


def run_cmd(cmd, gpu_id):
    def func():
        results = []
        base_cmd = cmd

        # 设置环境变量，确保基准测试也使用相同的GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_id

        print(f"Running custom command: {cmd} on GPU {gpu_id}")
        for i in tqdm(num_clients, desc="Custom Command"):
            total_number = 5000 if i == 1 else 20000
            cmd_new = base_cmd + [
                "--total_number", str(total_number),
                "--client", str(i),
            ]
            print(f"\nRunning command for {i} clients:")
            print(" ".join(cmd_new))

            result = subprocess.run(
                cmd_new,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                env=env  # 传递环境变量
            )

            print(f"Result for {i} clients:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)

            parsed_result = parse_result(result.stdout)
            results.append(parsed_result)
            print(f"Parsed result: {parsed_result}")

        return results

    return func


DEFAULT_PARAMS = [
    "--preprocess-instances", "8",
    "--max", "5",
    "--trt_instance_num", "2",
    "--timeout", "2",
]

TEST = {
    'triton_resnet_process': "tritonserver --model-repository=./model_repository/resnet/",
    'triton_resnet_thread': "tritonserver --model-repository=./model_repository/resnet/",
    'ensemble_dali_resnet_cpu': "tritonserver --model-repository=./model_repository/en_dalicpu/",
    'ensemble_dali_resnet_gpu': "tritonserver --model-repository=./model_repository/en_daligpu/"
}

if __name__ == "__main__":
    file = 'exp_w_cpu_gpu_results.json'

    results = []
    final_json = {}
    print(f"Starting benchmark with {len(TEST)} target(s)")

    # 获取要使用的GPU ID
    gpu_id = args.gpu_id

    for key, back_cmd in tqdm(TEST.items(), desc="Overall Progress"):
        print(f"\nStarting {key}...")

        # 启动Triton服务器
        server_process = start_triton_server(back_cmd, gpu_id)

        try:
            # 运行基准测试
            result = run_cmd(['python3', './benchmark.py',
                              '--model', key] + DEFAULT_PARAMS, gpu_id)()
            final_json[key] = result
            print(f"Completed {key}")
        except Exception as e:
            print(f"Error during benchmark for {key}: {e}")
        finally:
            # 停止Triton服务器
            stop_triton_server(server_process)
            # 等待一段时间确保端口释放
            time.sleep(5)

    import json
    with open(file, "w") as f:
        json.dump(final_json, f, indent=4)

    print("final result saved in ", file)
    print("\nFinal Results Summary:")
    for key, value in final_json.items():
        print(f"{key}:")
        for i, res in zip(num_clients, value):
            print(f"  {i} clients: {res}")
