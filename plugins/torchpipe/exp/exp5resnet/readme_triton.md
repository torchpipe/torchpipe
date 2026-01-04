
## experiment with Omniback
```bash

```
### Evaluation Section
- Prepare environment
```bash

# clone code 
git clone -b v1 ...
cd torchpipe/ && git submodule update --init --recursive

### ours => A10: ~/paper/v1/torchpipe/

# docker
img_name=nvcr.io/nvidia/tritonserver:25.05-py3 # triton 2.58.0


docker pull $img_name

docker run --name=exp_triton --runtime=nvidia --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

# install timm
# apt-get update && apt-get install -y cmake ninja-build

### optional: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# pip install -r plugins/torchpipe/exp/requirements.txt
ln -s /usr/bin/python3 /usr/bin/python

cd /workspace/plugins/torchpipe/exp
export CUDA_VISIBLE_DEVICES=0


/usr/src/tensorrt/bin/trtexec --onnx=resnet101.onnx --saveEngine=./resnet101_bs5i1.trt  --minShapes=input:1x3x224x224 --optShapes=input:5x3x224x224 --maxShapes=input:5x3x224x224 --fp16

cp resnet101_bs5i1.trt ./model_repository/resnet/resnet_trt/1/model.plan
cp resnet101_bs5i1.trt ./model_repository/cpu_en/resnet_trt/1/model.plan
cp resnet101_bs5i1.trt ./model_repository/en_dalicpu/resnet_trt/1/model.plan
cp resnet101_bs5i1.trt ./model_repository/en_daligpu/resnet_trt/1/model.plan

 pip install opencv-python-headless~=4.5 tritonclient[grpc]  psutil nvidia-ml-py requests


 tritonserver --model-repository=./model_repository/resnet/ 



### optional: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


# python3 ./benchmark.py  --model triton_resnet_process  --total_number 10000  --client 20
 python triton_w_cpu_gpu.py


```


- Triton
```bash
export CUDA_VISIBLE_DEVICES=0
 tritonserver --model-repository=./model_repository/cpu_en/  

 python3 benchmark.py --model ensemble_py_resnet --total_number 20000 --client 20 
```


 tritonserver --model-repository=./model_repository/en 

python3 ./benchmark.py --model ensemble_dali_resnet --total_number 20000 --client 20 


 ------
cd /workspace/plugins/torchpipe/exp
export CUDA_VISIBLE_DEVICES=0
 tritonserver --model-repository=./model_repository/cpu_en 

python3 decouple_eval/benchmark.py --model ensemble_py_resnet \
 --total_number 20000 --client 20 
```

- Triton Ensem. w/ GPU-dali
```bash
cd /workspace/plugins/torchpipe/exp
 tritonserver --model-repository=./model_repository/en 

python3 decouple_eval/benchmark.py --model ensemble_dali_resnet \
 --total_number 20000 --client 20 
```

https://github.com/NVIDIA/DALI/issues/4581   disable antialias

https://github.com/NVIDIA/DALI/issues/4581#issuecomment-1386888761

https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_6-building_complex_pipelines
 



 ## vllm
 img_name=nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 # triton 2.58.0
 docker stop exp_vllm && docker rm exp_vllm

docker run --name=exp_vllm --runtime=nvidia -e LC_ALL=C -e LANG=C  --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

```
### optional: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
 pip install pandas datasets
```

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server -tp 1 -pp 1 --gpu-memory-utilization 0.95         --model ./Llama-2-7b-chat-hf/ --port 8000 --disable-log-stats --disable-log-requests 

 python3 ./benchmark_throughput.py  --backend vllm      --model  Llama-2-7b-chat-hf        --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json    --num-prompts 200

python3 -c "import subprocess;subprocess.check_output(['/sbin/ldconfig', '-p']).decode()"

python3 -c "import subprocess;print(subprocess.check_output(['/sbin/ldconfig', '-p']).decode('latin-1'))"

/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py
24 > libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode('latin-1')