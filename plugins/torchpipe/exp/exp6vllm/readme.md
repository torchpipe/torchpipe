## vllm



 img_name=nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 # triton 2.58.0
 docker stop exp_vllm && docker rm exp_vllm

docker run --name=exp_vllm --runtime=nvidia -e LC_ALL=C -e LANG=C  --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

```
### optional: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
 pip install pandas datasets
```

### download model params:
assume we put it in ./Llama-2-7b-chat-hf/


### start vllm server
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server -tp 1 -pp 1 --gpu-memory-utilization 0.93         --model ./Llama-2-7b-chat-hf/ --port 8000 --disable-log-stats --disable-log-requests 

### test
```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


git clone -b v0.8.4 https://github.com/vllm-project/vllm.git


  python3 vllm/benchmark_serving.py         --backend vllm         --model ./Llama-2-7b-chat-hf/         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8000         --save-result         --result-dir results/         --result-filename vllm_llama7B_tp1_qps_2.json         --request-rate 2
  ```