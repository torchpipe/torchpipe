import os
import tempfile
import pickle
import subprocess
import argparse

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
    for i in num_clients:
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

        try:
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
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise  # 重新抛出异常，让外部处理

    return files


def run_cpu_preprocess_cmd():
    files = []
    print("Running CPU preprocess commands...")
    for i in num_clients:
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

        try:
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
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise  # 重新抛出异常，让外部处理

    return files


def run_cmd(cmd):
    def func():
        results = []
        base_cmd = cmd.strip().split()
        print(f"Running custom command: {cmd}")
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
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
                raise  # 重新抛出异常，让外部处理

        return results

    func.__name__ = cmd.strip().split(" ")[-1]
    return func


if __name__ == "__main__":

    file = 'omniback_cpu_gpu_results.json'
    targets = [run_cpu_preprocess_cmd, run_gpu_preprocess_cmd]
    if args.cmd:
        targets = [run_cmd(args.cmd)]
    results = []

    print(f"Starting benchmark with {len(targets)} target(s)")
    for func in targets:
        print(f"\nStarting {func.__name__}...")
        result = func()
        results.append(result)
        print(f"Completed {func.__name__}")
        # 打印中间汇总结果
        print(f"Intermediate results for {func.__name__}:")
        for i, res in zip(num_clients, result):
            print(f"  {i} clients: {res}")

    final_json = {}
    for k, v in zip(targets, results):
        final_json["omniback"+k.__name__] = v

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
