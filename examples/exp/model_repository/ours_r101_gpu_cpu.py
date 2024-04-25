import os
import tempfile, pickle


num_clients = [1, 5, 10, 20, 30, 40]


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
        cmd = f" cd /workspace/examples/exp && python3 decouple_eval/benchmark.py  --model resnet101  --preprocess gpu --preprocess-instances 4  --max 8 --trt_instance_num 2  --timeout 2 --total_number {total_number}  --client {i} --save {fi}"

        os.system(cmd)
        files.append(fi)
    return files


def run_cpu_preprocess_cmd():
    files = []
    for i in num_clients:
        # generate a temp file with '.pkl' extension
        fi = tempfile.mkstemp(suffix=".pkl")[1]
        total_number = 5000 if i == 1 else 20000
        cmd = f" cd /workspace/examples/exp &&python3 decouple_eval/benchmark.py --model resnet101  --preprocess-instances 7 --max 8 --trt_instance_num 2 --timeout 2  --total_number {total_number}  --client {i} --save {fi}"

        os.system(cmd)
        files.append(fi)
    return files


target = [run_cpu_preprocess_cmd, run_gpu_preprocess_cmd]
results = []
for func in target:
    files = func()
    results.append(read_result(files))

for k, v in zip(target, results):
    print(k, v)
