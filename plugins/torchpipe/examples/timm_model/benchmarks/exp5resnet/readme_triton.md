
## export onnx model
```bash
img_name=nvcr.io/nvidia/pytorch:25.05-py3 # triton 2.58.0
docker run --name=tmp --rm --runtime=nvidia --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

cd examples/timm_model/
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install fire cmake ninja timm onnxsim 

export MODEL=eva02_base_patch14_448.mim_in22k_ft_in22k_in1k
export HF_ENDPOINT=https://hf-mirror.com
python export.py --model_name=$MODEL --opset=20
```

## triton
```bash
 docker stop timm_triton && docker rm timm_triton

img_name=nvcr.io/nvidia/tritonserver:25.05-py3 # triton 2.58.0

docker run --name=timm_triton --runtime=nvidia --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

ln -s /usr/bin/python3 /usr/bin/python


pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
 
 pip install fire opencv-python-headless~=4.5 tritonclient[grpc]  psutil nvidia-ml-py requests
```
 

prepare trt engine
```bash
cd examples/timm_model/
export CUDA_VISIBLE_DEVICES=0

model_name=eva02_base_patch14_448.mim_in22k_ft_in22k_in1k
/usr/src/tensorrt/bin/trtexec --onnx=${model_name}.onnx --saveEngine=./${model_name}_bs4i1.trt  --minShapes=input:1x3x448x448 --optShapes=input:4x3x448x448 --maxShapes=input:4x3x448x448 --fp16



MODEL_REP=benchmarks/exp5resnet/model_repository
cp ./${model_name}_bs4i1.trt $MODEL_REP/resnet/resnet_trt/1/model.plan
cp ./${model_name}_bs4i1.trt $MODEL_REP/cpu_en/resnet_trt/1/model.plan
cp ./${model_name}_bs4i1.trt $MODEL_REP/en_dalicpu/resnet_trt/1/model.plan
cp ./${model_name}_bs4i1.trt $MODEL_REP/en_daligpu/resnet_trt/1/model.plan
```

- start triton server 
```bash
export CUDA_VISIBLE_DEVICES=0

cd benchmarks/exp5resnet/
#  python triton_w_cpu_gpu.py --num_clients=1,2,5,10
nohup python triton_w_cpu_gpu.py --num_clients=1,2,5,10 > log.txt 2>&1 &

#  tritonserver --model-repository=./model_repository/resnet/ \
#   --grpc-port=8003 \
#   --http-port=8004 \
#   --metrics-port=8005
```


### optional: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


# python3 ./benchmark.py  --model triton_resnet_process  --total_number 10000  --client 20
 python triton_w_cpu_gpu.py --num_clients=1,2,5,10


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