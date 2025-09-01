## vllm



 img_name=nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 # triton 2.58.0
 docker stop exp_vllm && docker rm exp_vllm

docker run --name=exp_vllm --runtime=nvidia -e LC_ALL=C -e LANG=C  --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

```
### optional: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
 pip install pandas datasets
```

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server -tp 1 -pp 1 --gpu-memory-utilization 0.95         --model ./Llama-2-7b-chat-hf/ --port 8000 --disable-log-stats --disable-log-requests 