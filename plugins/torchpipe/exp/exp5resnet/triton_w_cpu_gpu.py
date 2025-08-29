import os
import tempfile
import pickle
import subprocess  # 新增导入
import argparse
from tqdm import tqdm  # 用于进度条

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
args = parser.parse_args()

num_clients = [int(x.strip()) for x in args.num_clients.split(",")]


def parse_result(result):
    return result.strip().split("-------------------------------------------------------------------")[1]


def run_gpu_preprocess_cmd():
    files = []
    print("Running GPU preprocess commands...")
    # 添加进度条
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

        # 打印中间结果
        print(f"Result for {i} clients:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)

        parsed_result = parse_result(result.stdout)
        files.append(parsed_result)
        print(f"Parsed result: {parsed_result}")

    return files


def run_cpu_preprocess_cmd():
    files = []
    print("Running CPU preprocess commands...")
    # 添加进度条
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

        # 打印中间结果
        print(f"Result for {i} clients:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)

        parsed_result = parse_result(result.stdout)
        files.append(parsed_result)
        print(f"Parsed result: {parsed_result}")

    return files


def run_cmd(cmd):
    def func():
        results = []
        base_cmd = cmd
        print(f"Running custom command: {cmd}")
        # 添加进度条
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
                check=True
            )

            # 打印中间结果
            print(f"Result for {i} clients:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)

            parsed_result = parse_result(result.stdout)
            results.append(parsed_result)
            print(f"Parsed result: {parsed_result}")

        return results

    # func.__name__ = cmd.strip().split(" ")[-1]
    return func




TEST = {'triton_resnet_process':
        [
            "--preprocess-instances", "8",
            "--max", "5",
            "--trt_instance_num", "2",
            "--timeout", "2",
        ],
        'triton_resnet_thread':
        [
            "--preprocess-instances", "8",
            "--max", "5",
            "--trt_instance_num", "2",
            "--timeout", "2",
        ]
    }

if __name__ == "__main__":

    file = 'exp_w_cpu_gpu_results.json'
    
    results = []
    final_json = {}
    print(f"Starting benchmark with {len(TEST)} target(s)")
    # 添加总体进度条
    for key, cmd in tqdm(TEST.items(), desc="Overall Progress"):
        print(f"\nStarting {key}...")
        result = run_cmd(['python3', './benchmark.py', '--model', key] + cmd)()
        # results.append(result)
        final_json[key] = result
        print(f"Completed {key}")



    import json
    with open(file, "a") as f:
        json.dump(final_json, f, indent=4)

    print("final result saved in ", file)
    # 打印最终结果摘要
    print("\nFinal Results Summary:")
    for key, value in final_json.items():
        print(f"{key}:")
        for i, res in zip(num_clients, value):
            print(f"  {i} clients: {res}")
