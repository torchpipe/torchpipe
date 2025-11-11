
##  prepare three docker containers: vllm, omniback, benchmark clients
```bash

# server
img_name=nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 
docker run --name=exp_vllm --runtime=nvidia -e LC_ALL=C -e LANG=C  --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -itd $img_name /bin/bash


cd torchpipe/
img_name=nvcr.io/nvidia/pytorch:25.05-py3
docker run --name=exp_omniback --runtime=nvidia --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -itd $img_name /bin/bash

# client for benchmark
git clone -b v0.8.4 https://github.com/vllm-project/vllm.git
cd vllm
img_name=nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 
docker run --name=exp_vllmclient --runtime=nvidia -e LC_ALL=C -e LANG=C  --ipc=host  --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -itd $img_name /bin/bash
```



## start vllm server
```bash
docker exec -it exp_vllm bash
rm -rf /opt/hpcx/ncclnet_plugin && ldconfig

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server -tp 1 -pp 1 --gpu-memory-utilization 0.93       --port 8001 --disable-log-stats --disable-log-requests --served-model-name llama2  --model meta-llama/Llama-2-7b-chat-hf # --model Llama-2-7b-chat-hf/
```

## export onnx for omniback

```bash
docker exec -it exp_omniback bash
rm -rf /opt/hpcx/ncclnet_plugin && ldconfig

rm -rf dist/*.whl && python setup.py bdist_wheel && pip install dist/*.whl
cd plugins/torchpipe/ && rm -rf dist/*.whl && python setup.py bdist_wheel && pip install dist/*.whl

cd /workspace/plugins/torchpipe/exp/exp6vllm/

# pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install -r requirements.txt 
python3 ../../examples/llama2/models/export_onnx_v2.py --num_layers 32 --model_id meta-llama/Llama-2-7b-chat-hf # # --model_id  path_to/Llama-2-7b-chat-hf/

ls -alh exported_params/

# onnx2tensorrt
python ../../examples/llama2/plain_llama2.py --num_layers=32
```
 
 ## start omniback server
```bash

# for A10-24G
python ../../examples/llama2/streaming_llama2.py --num_layers=32 --port=8000 --max_num_page=1024 

```


## clients
```bash
docker exec -it exp_vllmclient bash
rm -rf /opt/hpcx/ncclnet_plugin && ldconfig

# pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pip install pandas datasets

mkdir results

export MODEL_ID=meta-llama/Llama-2-7b-chat-hf
# or export MODEL_ID=path_to/Llama-2-7b-chat-hf
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

sh all_clients.sh  && sh all_clients.sh

# python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8000         --save-result         --result-dir results/         --result-filename omniback_llama7B_tp1_qps_2.json         --request-rate 2

# python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8000         --save-result         --result-dir results/         --result-filename omniback_llama7B_tp1_qps_3.json         --request-rate 3

# python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8001         --save-result         --result-dir results/         --result-filename vllm_llama7B_tp1_qps_2.json         --request-rate 2 --served-model-name llama2 

# python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8001         --save-result         --result-dir results/         --result-filename vllm_llama7B_tp1_qps_3.json         --request-rate 3 --served-model-name llama2 


```
 
##  result
```
============ Serving Benchmark Result ============
Successful requests:                     500
Benchmark duration (s):                  275.42
Total input tokens:                      117316
Total generated tokens:                  105674
Request throughput (req/s):              1.82
Output token throughput (tok/s):         383.69
Total Token throughput (tok/s):          809.64
---------------Time to First Token----------------
Mean TTFT (ms):                          110.45
Median TTFT (ms):                        80.43
P99 TTFT (ms):                           333.85
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          43.40
Median TPOT (ms):                        43.00
P99 TPOT (ms):                           61.47
---------------Inter-token Latency----------------
Mean ITL (ms):                           43.53
Median ITL (ms):                         39.76
P99 ITL (ms):                            167.74
==================================================

============ Serving Benchmark Result ============
Successful requests:                     500
Benchmark duration (s):                  194.65
Total input tokens:                      117316
Total generated tokens:                  106180
Request throughput (req/s):              2.57
Output token throughput (tok/s):         545.49
Total Token throughput (tok/s):          1148.19
---------------Time to First Token----------------
Mean TTFT (ms):                          1495.44
Median TTFT (ms):                        174.39
P99 TTFT (ms):                           8502.62
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          57.38
Median TPOT (ms):                        57.33
P99 TPOT (ms):                           84.89
---------------Inter-token Latency----------------
Mean ITL (ms):                           57.76
Median ITL (ms):                         50.43
P99 ITL (ms):                            225.48
==================================================
============ Serving Benchmark Result ============
Successful requests:                     500
Benchmark duration (s):                  275.86
Total input tokens:                      117316
Total generated tokens:                  105642
Request throughput (req/s):              1.81
Output token throughput (tok/s):         382.96
Total Token throughput (tok/s):          808.24
---------------Time to First Token----------------
Mean TTFT (ms):                          116.48
Median TTFT (ms):                        86.02
P99 TTFT (ms):                           335.25
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          43.93
Median TPOT (ms):                        43.09
P99 TPOT (ms):                           65.04
---------------Inter-token Latency----------------
Mean ITL (ms):                           43.35
Median ITL (ms):                         39.86
P99 ITL (ms):                            176.02
==================================================
============ Serving Benchmark Result ============
Successful requests:                     500
Benchmark duration (s):                  199.78
Total input tokens:                      117316
Total generated tokens:                  105853
Request throughput (req/s):              2.50
Output token throughput (tok/s):         529.84
Total Token throughput (tok/s):          1117.05
---------------Time to First Token----------------
Mean TTFT (ms):                          3523.20
Median TTFT (ms):                        386.31
P99 TTFT (ms):                           14072.71
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          59.49
Median TPOT (ms):                        57.27
P99 TPOT (ms):                           116.30
---------------Inter-token Latency----------------
Mean ITL (ms):                           57.06
Median ITL (ms):                         50.75
P99 ITL (ms):                            217.70
==================================================
```

     