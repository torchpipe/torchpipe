import os
import tempfile, pickle


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
    default="1, 10, 20, 30",
    help="basic num_clients",
)
args = parser.parse_args()

num_clients = [int(x.strip()) for x in args.num_clients.split(",")]


def read_result(files):
    result = []
    for i in files:
        data = pickle.load(open(i, "rb"))
        result.append(data)
    return result


def run_gpu_preprocess_cmd():
    files = []
    for i in num_clients:
        # generate a temp file with '.pkl' extension
        fi = tempfile.mkstemp(suffix=".pkl")[1]
        total_number = 5000 if i == 1 else 20000
        cmd = f"python3 decouple_eval/benchmark.py  --model resnet101  --preprocess gpu --preprocess-instances 8  --max 8 --trt_instance_num 2  --timeout 4 --total_number {total_number}  --client {i} --save {fi}"

        os.system(cmd)
        files.append(fi)
    return files


def run_cpu_preprocess_cmd():
    files = []
    for i in num_clients:
        # generate a temp file with '.pkl' extension
        fi = tempfile.mkstemp(suffix=".pkl")[1]
        total_number = 5000 if i == 1 else 20000
        cmd = f"python3 decouple_eval/benchmark.py --model resnet101  --preprocess-instances 8 --max 8 --trt_instance_num 2 --timeout 4  --total_number {total_number}  --client {i} --save {fi}"

        os.system(cmd)
        files.append(fi)
    return files


def run_cmd(cmd):
    def func():
        files = []
        for i in num_clients:
            # generate a temp file with '.pkl' extension
            fi = tempfile.mkstemp(suffix=".pkl")[1]
            total_number = 5000 if i == 1 else 20000
            cmd_new = f"{cmd} --total_number {total_number}  --client {i} --save {fi}"
            print("cmd: ", cmd_new)
            os.system(cmd_new)
            files.append(fi)
        return files

    func.__name__ = cmd.strip().split(" ")[-1]
    return func


targets = [run_cpu_preprocess_cmd, run_gpu_preprocess_cmd]
if args.cmd:
    targets = [run_cmd(args.cmd)]
results = []
for func in targets:
    files = func()
    results.append(read_result(files))

for k, v in zip(targets, results):
    print(k.__name__ + " = ", v)
