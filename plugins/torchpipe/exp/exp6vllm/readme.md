
##  prepare three docker containers: vllm, hami, benchmark clients
```bash

# server
img_name=nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 
docker run --name=exp_vllm --runtime=nvidia -e LC_ALL=C -e LANG=C  --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -itd $img_name /bin/bash


cd torchpipe/
img_name=nvcr.io/nvidia/pytorch:25.05-py3
docker run --name=exp_hami --runtime=nvidia --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -itd $img_name /bin/bash

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

## export onnx for hami

```bash
docker exec -it exp_hami bash
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
 
 ## start hami server
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

sh all_clients.sh

sh all_clients.sh

# python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8000         --save-result         --result-dir results/         --result-filename hami_llama7B_tp1_qps_2.json         --request-rate 2

# python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8000         --save-result         --result-dir results/         --result-filename hami_llama7B_tp1_qps_3.json         --request-rate 3

# python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8001         --save-result         --result-dir results/         --result-filename vllm_llama7B_tp1_qps_2.json         --request-rate 2 --served-model-name llama2 

# python3 benchmarks/benchmark_serving.py         --backend vllm         --model $MODEL_ID         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8001         --save-result         --result-dir results/         --result-filename vllm_llama7B_tp1_qps_3.json         --request-rate 3 --served-model-name llama2 


```
 
##  result
```
============ Serving Benchmark Result ============
Successful requests:                     500
Benchmark duration (s):                  273.85
Total input tokens:                      117316
Total generated tokens:                  105804
Request throughput (req/s):              1.83
Output token throughput (tok/s):         386.35
Total Token throughput (tok/s):          814.74
---------------Time to First Token----------------
Mean TTFT (ms):                          131.95
Median TTFT (ms):                        84.91
P99 TTFT (ms):                           776.14
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.34
Median TPOT (ms):                        43.71
P99 TPOT (ms):                           67.25
---------------Inter-token Latency----------------
Mean ITL (ms):                           44.31
Median ITL (ms):                         40.01
P99 ITL (ms):                            169.12
==================================================


============ Serving Benchmark Result ============
Successful requests:                     500
Benchmark duration (s):                  195.00
Total input tokens:                      117316
Total generated tokens:                  105106
Request throughput (req/s):              2.56
Output token throughput (tok/s):         539.02
Total Token throughput (tok/s):          1140.65
---------------Time to First Token----------------
Mean TTFT (ms):                          1543.46
Median TTFT (ms):                        171.70
P99 TTFT (ms):                           8008.28
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          58.56
Median TPOT (ms):                        57.12
P99 TPOT (ms):                           105.02
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.42
Median ITL (ms):                         50.28
P99 ITL (ms):                            222.65
==================================================
```

     