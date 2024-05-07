import os
import tempfile, pickle


num_clients = [1, 5, 10, 20, 30]

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cmd",
    dest="cmd",
    type=str,
    default="",
    help="basic cmd",
)
args = parser.parse_args()


def read_result(files):
    result = []
    for i in files:
        data = pickle.load(open(i, "rb"))
        result.append(data)
    return result


def run_multi_process_cmd():
    files = []
    for i in num_clients:
        # generate a temp file with '.pkl' extension
        fi = tempfile.mkstemp(suffix=".pkl")[1]
        total_number = 5000 if i == 1 else 20000
        cmd = f"USE_PROCESS=1 python3 decouple_eval/benchmark.py   --model triton_resnet  --total_number {total_number}  --client {i} --save {fi}"

        os.system(cmd)
        files.append(fi)
    return files


def run_multi_thread_cmd():
    files = []
    for i in num_clients:
        # generate a temp file with '.pkl' extension
        fi = tempfile.mkstemp(suffix=".pkl")[1]
        total_number = 5000 if i == 1 else 20000
        cmd = f"python3 decouple_eval/benchmark.py --model triton_resnet  --total_number {total_number}  --client {i} --save {fi}"

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

            os.system(cmd_new)
            files.append(fi)
        return files

    func.__name__ = cmd.strip().split(" ")[-1]
    return func


targets = [run_multi_thread_cmd, run_multi_process_cmd]
if args.cmd:
    targets = [run_cmd(args.cmd)]
results = []
for func in targets:
    files = func()
    results.append(read_result(files))

for k, v in zip(targets, results):
    print(k.__name__ + " = ", v)
