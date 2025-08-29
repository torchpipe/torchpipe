import os
import tempfile
import pickle
import subprocess
import argparse
from tqdm import tqdm
import time
import signal
import requests
import json

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
parser.add_argument(
    "--triton_http_port",
    dest="triton_http_port",
    type=int,
    default=8000,
    help="Triton HTTP port",
)
args = parser.parse_args()

num_clients = [int(x.strip()) for x in args.num_clients.split(",")]


def parse_result(result):
    return result.strip().split("-------------------------------------------------------------------")[1]


def check_triton_ready(port, max_retries=30, retry_interval=2):
    """检查Triton服务器是否已准备好"""
    health_url = f"http://localhost:{port}/v2/health/ready"

    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print("Triton server is ready")
                return True
        except requests.exceptions.RequestException:
            pass

        if i < max_retries - 1:
            print(
                f"Waiting for Triton server to be ready... (attempt {i+1}/{max_retries})")
            time.sleep(retry_interval)

    print("Triton server failed to become ready within the expected time")
    return False


def check_triton_stopped(port, max_retries=5, retry_interval=1):
    """验证Triton服务器是否已停止且端口不再可用"""
    health_url = f"http://localhost:{port}/v2/health/ready"

    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=3)
            # 如果还能收到响应，说明服务器仍在运行
            print(
                f"Triton server still responding (attempt {i+1}/{max_retries})")
        except requests.exceptions.ConnectionError:
            # 连接被拒绝，说明服务器已停止
            print("Triton server has stopped and port is no longer accessible")
            return True
        except requests.exceptions.RequestException:
            # 其他异常也视为服务器已停止
            print("Triton server has stopped (connection failed)")
            return True

        time.sleep(retry_interval)

    print("Triton server might still be running or port is still in use")
    return False


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

    # 等待服务器启动，通过健康检查确认
    if check_triton_ready(args.triton_http_port):
        print("Triton server started successfully")
    else:
        print("Triton server failed to start")
        stop_triton_server(process)
        raise RuntimeError("Triton server failed to start")

    return process


def stop_triton_server(process):
    """停止Triton服务器进程并验证"""
    if process:
        try:
            # 终止整个进程组
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=15)
            print("Triton server stopped successfully")
        except subprocess.TimeoutExpired:
            print("Triton server did not stop gracefully, forcing termination")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            print("Triton server process already terminated")
        except Exception as e:
            print(f"Error stopping Triton server: {e}")

        # 验证服务器是否真正停止
        check_triton_stopped(args.triton_http_port)


def run_cmd(cmd, gpu_id, pbar=None):
    def func():
        results = []
        base_cmd = cmd

        # 设置环境变量，确保基准测试也使用相同的GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_id

        print(f"Running custom command: {cmd} on GPU {gpu_id}")
        for i in num_clients:
            total_number = 5000 if i == 1 else 20000
            cmd_new = base_cmd + [
                "--total_number", str(total_number),
                "--client", str(i),
            ]
            print(f"\nRunning command for {i} clients:")
            print(" ".join(cmd_new))

            try:
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

            except subprocess.CalledProcessError as e:
                print(f"Command failed with exit code {e.returncode}")
                print(f"Command: {e.cmd}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
                # 将错误信息添加到结果中，以便后续分析
                results.append(f"ERROR: {e.stderr}")
                # 继续执行下一个客户端数量
                continue

            # 更新总体进度条
            if pbar:
                pbar.update(1)
                pbar.set_description(f"Completed {i} clients for current test")

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
    file = 'triton_cpu_gpu_results.json'

    results = []
    final_json = {}

    # 获取要使用的GPU ID
    gpu_id = args.gpu_id

    # 计算总任务数
    total_tasks = len(TEST) * len(num_clients)

    # 创建总体进度条
    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        for key, back_cmd in TEST.items():
            pbar.set_description(f"Starting {key}")
            print(f"\nStarting {key}...")

            # 启动Triton服务器
            server_process = start_triton_server(back_cmd, gpu_id)

            try:
                # 运行基准测试，传递进度条对象
                result = run_cmd(['python3', './benchmark.py',
                                  '--model', key] + DEFAULT_PARAMS, gpu_id, pbar)()
                final_json[key] = result
                print(f"Completed {key}")
            except Exception as e:
                print(f"Error during benchmark for {key}: {e}")
                # 出错时也更新进度条
                pbar.update(len(num_clients))
            finally:
                # 停止Triton服务器
                stop_triton_server(server_process)
                # 等待一段时间确保端口释放
                time.sleep(2)

    with open(file, "a") as f:
        json.dump(final_json, f, indent=4)

    print("\nfinal result saved in ", file)
    print("\nFinal Results Summary:")
    for key, value in final_json.items():
        print(f"{key}:")
        for i, res in zip(num_clients, value):
            print(f"  {i} clients: {res}")
