import os
import tempfile
import pickle
import subprocess  # 新增导入
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



def run_gpu_preprocess_cmd():
    files = []
    for i in num_clients:
        total_number = 5000 if i == 1 else 20000
        cmd = [
            "python3",
            "./benchmark.py",
            "--model", "resnet101",
            "--preprocess", "gpu",
            "--preprocess-instances", "8",
            "--max", "8",
            "--trt_instance_num", "2",
            "--timeout", "2",
            "--total_number", str(total_number),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        files.append(result.stdout.strip().split('\n'))

    return files


def run_cpu_preprocess_cmd():
    files = []
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

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        files.append(result.stdout.strip().split('\n'))

    return files


def run_cmd(cmd):
    def func():
        results = []
        base_cmd = cmd.strip().split()
        for i in num_clients:
            total_number = 5000 if i == 1 else 20000
            cmd_new = base_cmd + [
                "--total_number", str(total_number),
                "--client", str(i),
            ]
            print("cmd: ", " ".join(cmd_new))

            result = subprocess.run(
                cmd_new,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            results.append(result)

        return results

    func.__name__ = cmd.strip().split(" ")[-1]
    return func


if __name__ == "__main__":

    file = 'hami_w_cpu_gpu_results.json'
    targets = [run_cpu_preprocess_cmd, run_gpu_preprocess_cmd]
    if args.cmd:
        targets = [run_cmd(args.cmd)]
    results = []
    for func in targets:
        result = func()
        results.append(result)

    final_json = {}
    for k, v in zip(targets, results):
        final_json[k.__name__] = v
    import json 
    with open(file, "w") as f:
        json.dump(final_json, f, indent=4)
    print("final result saved in ", file)