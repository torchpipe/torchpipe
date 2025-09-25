
## triton client with multiple theads & triton client with multiple process

- start triton server
```bash
cd paper/torchpipe/
img_name=nvcr.io/nvidia/tritonserver:25.05-py3
nvidia-docker run --name=triton_exp -it --rm --cpus=8 --network=host --runtime=nvidia --privileged  --name triton8 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/workspace   $img_name bash

cd /workspace/examples/exp

onnx_path=./resnet101.onnx # please exported with dynamic batch size. refer to model_repository/en/export_onnx.py
/usr/src/tensorrt/bin/trtexec --onnx=$onnx_path --saveEngine=./model_repository/resnet/resnet_trt/1/model.plan --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224 --fp16
#  --explicitBatch
export CUDA_VISIBLE_DEVICES=3
tritonserver --model-repository=./model_repository/resnet
```


- test with  multiple theads clients

```bash
docker exec -it triton8 bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install tritonclient[grpc] py3nvml nvidia-ml-py psutil opencv-python-headless

cd /workspace/examples/exp

ln -s /usr/bin/python3 /usr/bin/python
CUDA_VISIBLE_DEVICES=3 python model_repository/r101_triton_mt_mp.py
```

- result
```python
 
run_multi_thread_cmd =  [{1: {'QPS': 103.0, 'TP50': 9.68, 'TP99': 10.0, 'GPU Usage': 18.0}}, {5: {'QPS': 457.23, 'TP50': 10.87, 'TP99': 11.88, 'GPU Usage': 51.0}}, {10: {'QPS': 586.39, 'TP50': 16.8, 'TP99': 24.66, 'GPU Usage': 61.0}}, {20: {'QPS': 530.82, 'TP50': 36.98, 'TP99': 56.1, 'GPU Usage': 57.0}}, {30: {'QPS': 522.35, 'TP50': 56.15, 'TP99': 89.72, 'GPU Usage': 56.0}}]
run_multi_process_cmd =  [{1: {'QPS': 106.08, 'TP50': 9.39, 'TP99': 9.85, 'GPU Usage': 19.0}}, {5: {'QPS': 447.1, 'TP50': 11.16, 'TP99': 11.69, 'GPU Usage': 22.0}}, {10: {'QPS': 839.92, 'TP50': 11.86, 'TP99': 13.07, 'GPU Usage': 35.0}}, {20: {'QPS': 1082.09, 'TP50': 13.4, 'TP99': 47.59, 'GPU Usage': 45.0}}, {30: {'QPS': 1046.92, 'TP50': 14.24, 'TP99': 70.52, 'GPU Usage': 42.0}}]
```

- return after step A and step B
```bash
# thread
cd /workspace/examples/exp && RETURN_A=1 python3 decouple_eval/benchmark.py --model triton_resnet  --total_number 10000 --client 20
# {20: {'QPS': 1622.29, 'TP50': 4.61, 'TP99': 71.73, 'GPU Usage': '-'}}
cd /workspace/examples/exp && RETURN_B=1 python3 decouple_eval/benchmark.py --model triton_resnet  --total_number 10000 --client 20
# {20: {'QPS': 1236.84, 'TP50': 15.91, 'TP99': 24.19, 'GPU Usage': '-'}}
cd /workspace/examples/exp && RETURN_C=1 python3 decouple_eval/benchmark.py --model triton_resnet  --total_number 10000 --client 20

# process
cd /workspace/examples/exp && RETURN_A=1 USE_PROCESS=1 python3 decouple_eval/benchmark.py --model triton_resnet  --total_number 20000 --client 20
# {20: {'QPS': 1457.24, 'TP50': 4.79, 'TP99': 74.68, 'GPU Usage': '-'}}
cd /workspace/examples/exp && RETURN_B=1 USE_PROCESS=1 python3 decouple_eval/benchmark.py --model triton_resnet  --total_number 20000 --client 20
# {20: {'QPS': 1278.52, 'TP50': 5.46, 'TP99': 72.65, 'GPU Usage': '-'}}
cd /workspace/examples/exp && RETURN_C=1 USE_PROCESS=1 python3 decouple_eval/benchmark.py --model triton_resnet  --total_number 20000 --client 20
# {20: {'QPS': 962.0, 'TP50': 10.32, 'TP99': 65.53, 'GPU Usage': 42.0}}
```

## triton ensemble
- prepare env
```bash
docke rexec -it triton8 bash
cd /workspace/examples/exp
```

- convert onnx to trt
```bash
onnx_path=./resnet101.onnx
# /usr/src/tensorrt/bin/trtexec --onnx=$onnx_path --saveEngine=./model_repository/en/resnet_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:4x3x224x224 --fp16

cp ./model_repository/resnet/resnet_trt/1/model.plan ./model_repository/en/resnet_trt/1/model.plan
# /usr/src/tensorrt/bin/trtexec --onnx=$onnx_path --saveEngine=./model_repository/en/resnet_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224 --fp16
```

- start
 
```bash
export CUDA_VISIBLE_DEVICES=3

tritonserver --model-repository=./model_repository/en 
```

- test
```bash
docke rexec -it triton8 bash
cd /workspace/examples/exp
python model_repository/r101_triton_mt_mp.py --cmd "python3 decouple_eval/benchmark.py  --model ensemble_dali_resnet "
```

- result
```python
ensemble_dali_resnet =  [{1: {'QPS': 184.37, 'TP50': 5.42, 'TP99': 5.53, 'GPU Usage': 99.0}}, {5: {'QPS': 849.02, 'TP50': 5.87, 'TP99': 6.36, 'GPU Usage': 99.0}}, {10: {'QPS': 1496.06, 'TP50': 6.29, 'TP99': 9.82, 'GPU Usage': 99.0}}, {20: {'QPS': 1952.91, 'TP50': 10.37, 'TP99': 13.45, 'GPU Usage': 98.0}}, {30: {'QPS': 2210.32, 'TP50': 12.78, 'TP99': 18.53, 'GPU Usage': 99.0}}]
```





## triton ensemble with cpu preprocess
- prepare env
```bash
docke rexec -it triton8 bash
cd /workspace/examples/exp
```

- convert onnx to trt
```bash
cp ./model_repository/resnet/resnet_trt/1/model.plan ./model_repository/cpu_en/resnet_trt/1/ 

```

- start
start 
```bash
CUDA_VISIBLE_DEVICES=3 tritonserver --model-repository=./model_repository/cpu_en 
```

- test
```bash
docker exec -it triton8 bash
cd /workspace/examples/exp
python model_repository/r101_triton_mt_mp.py --cmd "python3 decouple_eval/benchmark.py  --model ensemble_py_resnet "
```

- result
```python
ensemble_py_resnet [{1: {'QPS': 107.53, 'TP50': 9.27, 'TP99': 9.78, 'GPU Usage': 19.0}}, {5: {'QPS': 496.96, 'TP50': 10.01, 'TP99': 10.79, 'GPU Usage': 25.0}}, {10: {'QPS': 943.46, 'TP50': 10.53, 'TP99': 11.73, 'GPU Usage': 49.0}}, {20: {'QPS': 1218.09, 'TP50': 14.45, 'TP99': 32.01, 'GPU Usage': 74.0}}, {30: {'QPS': 1217.05, 'TP50': 21.52, 'TP99': 39.79, 'GPU Usage': 74.0}}, {40: {'QPS': 1211.35, 'TP50': 29.07, 'TP99': 47.22, 'GPU Usage': 75.0}}]
```


## (section 8)torchpipe with CPU preprocess and GPU preprocess
- environment
```bash
# start a docker with torchpie installed
img_name=hub.c.163.com/neteaseis/ai/torchpipe:0.4.2

cd paper/torchpipe/
nvidia-docker run -it   --cpus=8  --name a108 --network=host --runtime=nvidia --privileged  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/workspace   $img_name bash

cd examples/exp/ && pip install pynvml
```

- prepare tensorrt model from resnet101.onnx 
```bash
python decouple_eval/benchmark.py --model resnet101 --total_number 11000  --preprocess-instances 7 --max 8 --trt_instance_num 2 --timeout 2 --client 20
```

- test
```bash
CUDA_VISIBLE_DEVICES=3 python model_repository/ours_r101_gpu_cpu.py
```

- result
```python       
run_cpu_preprocess_cmd =  [{1: {'QPS': 197.33, 'TP50': 5.02, 'TP99': 7.06, 'GPU Usage': 34.0}}, {5: {'QPS': 657.11, 'TP50': 7.6, 'TP99': 7.72, 'GPU Usage': 30.0}}, {10: {'QPS': 1248.39, 'TP50': 7.99, 'TP99': 8.33, 'GPU Usage': 59.0}}, {20: {'QPS': 2117.64, 'TP50': 9.58, 'TP99': 14.93, 'GPU Usage': 93.0}}, {30: {'QPS': 2112.69, 'TP50': 13.07, 'TP99': 19.78, 'GPU Usage': 88.0}}, {40: {'QPS': 2125.79, 'TP50': 19.15, 'TP99': 25.69, 'GPU Usage': 88.0}}]
run_gpu_preprocess_cmd =  [{1: {'QPS': 351.91, 'TP50': 2.79, 'TP99': 4.83, 'GPU Usage': 64.0}}, {5: {'QPS': 922.36, 'TP50': 5.41, 'TP99': 5.51, 'GPU Usage': 50.0}}, {10: {'QPS': 1833.63, 'TP50': 5.42, 'TP99': 5.7, 'GPU Usage': 97.0}}, {20: {'QPS': 2511.63, 'TP50': 7.92, 'TP99': 8.71, 'GPU Usage': 100.0}}, {30: {'QPS': 2611.86, 'TP50': 11.38, 'TP99': 12.43, 'GPU Usage': 88.5}}, {40: {'QPS': 2585.19, 'TP50': 15.43, 'TP99': 16.32, 'GPU Usage': 98.5}}]
```

- return after step A and step B
```bash
python decouple_eval/benchmark.py  --model resnet101  --preprocess gpu --preprocess-instances 4  --max 8 --trt_instance_num 2  --timeout 2 --total_number 15000 --client 20
{20: {'QPS': 2547.91, 'TP50': 7.67, 'TP99': 12.04, 'GPU Usage': 100.0}}


python decouple_eval/benchmark.py  --model empty  --preprocess gpu --preprocess-instances 4  --max 8 --trt_instance_num 2  --timeout 2 --total_number 15000 --client 20
{20: {'QPS': 3965.89, 'TP50': 4.93, 'TP99': 5.1, 'GPU Usage': '-'}}

python decouple_eval/benchmark.py  --model empty  --preprocess cpu --preprocess-instances 7  --max 8 --trt_instance_num 2  --timeout 2 --total_number 15000 --client 20
{20: {'QPS': 2151.18, 'TP50': 9.35, 'TP99': 12.06, 'GPU Usage': '-'}}

```




## case stady

- environment
```bash
# start a docker with torchpie installed
img_name=hub.c.163.com/neteaseis/ai/torchpipe:0.4.2

nvidia-docker run -it --rm  --name a10 --network=host --runtime=nvidia --privileged  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/workspace   $img_name bash
```

- test
```bash
cd examples/exp/ && pip install pynvml 

export CUDA_VISIBLE_DEVICES=0 

python model_repository/ours_r101_gpu_cpu.py --cmd "python3 decouple_eval/benchmark.py   --preprocess-instances 14 --max 8 --trt_instance_num 5 --timeout 2 --model resnet101" --num_clients 1,3,5,8,10,20,40,80,160 

```

- result
```python  
```















# triton gpu

```

# /usr/src/tensorrt/bin/trtexec
onnx_path=./resnet101.onnx
CUDA_VISIBLE_DEVICES=3 /usr/src/tensorrt/bin/trtexec --onnx=$onnx_path --saveEngine=./model_repository/en/resnet_trt/1/model.plan_4 --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:4x3x224x224 --fp16
```

```
export CUDA_VISIBLE_DEVICES=0
tritonserver --model-repository=./model_repository





img_name=nvcr.io/nvidia/tritonserver:23.06-py3
 
 <!-- cd torchpipe -->

nvidia-docker run -it --rm --runtime=nvidia --privileged  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v `pwd`:/workspace   $img_name bash


nvidia-docker run -it --rm --cpus=8 --network=host --runtime=nvidia --privileged  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/workspace   $img_name bash


nvidia-docker run -it --rm --runtime=nvidia --privileged  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v `pwd`:/workspace   $img_name bash


 cd /workspace/
    mkdir  /workspace/model_repository/en/ensemble_dali_resnet/1            
CUDA_VISIBLE_DEVICES=3 tritonserver --model-repository=./model_repository/en 



```

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install tritonclient[grpc] pynvml psutil opencv-python-headless torch


# r101 ensemble
docker exec -it triton bash
python3 decouple_eval/benchmark.py  --model triton_resnet_ensemble   --total_number 30000 --client 21
# 2276.1 => {22: {'QPS': 2282.95, 'TP50': 9.61, 'TP99': 12.11, 'GPU Usage': 99.0}}
# {21: {'QPS': 2239.49, 'TP50': 9.36, 'TP99': 11.99, 'GPU Usage': 99.0}} 772MiB
# {30: {'QPS': 2529.0, 'TP50': 11.82, 'TP99': 13.31, 'GPU Usage': 99.0}}

# new 
# {1: {'QPS': 273.11, 'TP50': 3.64, 'TP99': 3.8, 'GPU Usage': 57.0}}
# {5: {'QPS': 1004.09, 'TP50': 5.08, 'TP99': 5.96, 'GPU Usage': 100.0}}
# {10: {'QPS': 1612.5, 'TP50': 6.12, 'TP99': 7.68, 'GPU Usage': 100.0}}
# {20: {'QPS': 2012.59, 'TP50': 10.0, 'TP99': 13.1, 'GPU Usage': 100.0}}
# {30: {'QPS': 2341.82, 'TP50': 12.39, 'TP99': 15.67, 'GPU Usage': 100.0}}
# {40: {'QPS': 2455.28, 'TP50': 16.17, 'TP99': 19.13, 'GPU Usage': 100.0}}
```

 



 # r101 C.P.  M.T./M.P.
```bash

cd paper/torchpipe/
img_name=nvcr.io/nvidia/tritonserver:23.06-py3
nvidia-docker run -it --rm --cpus=7 --network=host --runtime=nvidia --privileged  --name triton --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/workspace   $img_name bash


export CUDA_VISIBLE_DEVICES=3

cd /workspace/examples/exp &&  tritonserver --model-repository=./model_repository/resnet



docker exec -it triton bash

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install tritonclient[grpc] pynvml psutil opencv-python-headless



export CUDA_VISIBLE_DEVICES=3

 cd /workspace/examples/exp &&  python3 decouple_eval/benchmark.py --model triton_resnet  --total_number 10000 --client 20


# or CUDA_VISIBLE_DEVICES=3 python model_repository/r101_triton_mt_mp.py


# {1: {'QPS': 123.85, 'TP50': 8.07, 'TP99': 8.6, 'GPU Usage': 25.0}}
# {5: {'QPS': 501.03, 'TP50': 9.99, 'TP99': 11.63, 'GPU Usage': 86.5}}
# {10: {'QPS': 611.33, 'TP50': 15.61, 'TP99': 25.2, 'GPU Usage': 92.0}}
# {20: {'QPS': 558.62, 'TP50': 35.24, 'TP99': 52.74, 'GPU Usage': 95.0}} 
# {30: {'QPS': 518.59, 'TP50': 56.78, 'TP99': 90.04, 'GPU Usage': 92.0}}
# {40: {'QPS': 494.01, 'TP50': 78.96, 'TP99': 130.94, 'GPU Usage': 91.0}}
# .set_data_from_numpy
 
 cd /workspace/examples/exp && USE_PROCESS=1 python3 decouple_eval/benchmark.py  --model triton_resnet  --total_number 10000  --client 5


# new
# {1: {'QPS': 130.66, 'TP50': 7.64, 'TP99': 8.38, 'GPU Usage': 26.5}}
# {5: {'QPS': 599.57, 'TP50': 8.34, 'TP99': 9.03, 'GPU Usage': 78.0}}
# {10: {'QPS': 958.6, 'TP50': 10.66, 'TP99': 11.45, 'GPU Usage': 100.0}}
# {20: {'QPS': 934.84, 'TP50': 12.78, 'TP99': 59.06, 'GPU Usage': 59.0}}
# {30: {'QPS': 910.54, 'TP50': 14.9, 'TP99': 77.68, 'GPU Usage': 43.5}}
# {40: {'QPS': 877.57, 'TP50': 18.78, 'TP99': 89.13, 'GPU Usage': 34.0}}
```

[[{1: {'QPS': 102.78, 'TP50': 9.72, 'TP99': 10.08, 'GPU Usage': '-'}}, {5: {'QPS': 536.12, 'TP50': 8.99, 'TP99': 10.66, 'GPU Usage': '-'}}, {10: {'QPS': 993.64, 'TP50': 9.88, 'TP99': 11.5, 'GPU Usage': '-'}}, {20: {'QPS': 941.82, 'TP50': 11.41, 'TP99': 63.11, 'GPU Usage': '-'}}, {30: {'QPS': 899.98, 'TP50': 13.28, 'TP99': 79.37, 'GPU Usage': '-'}}, {40: {'QPS': 895.02, 'TP50': 17.64, 'TP99': 85.74, 'GPU Usage': '-'}}], [{1: {'QPS': 98.83, 'TP50': 10.09, 'TP99': 10.46, 'GPU Usage': '-'}}, {5: {'QPS': 440.64, 'TP50': 11.19, 'TP99': 12.41, 'GPU Usage': '-'}}, {10: {'QPS': 593.08, 'TP50': 17.0, 'TP99': 23.79, 'GPU Usage': '-'}}, {20: {'QPS': 568.18, 'TP50': 34.52, 'TP99': 51.83, 'GPU Usage': '-'}}, {30: {'QPS': 486.38, 'TP50': 58.19, 'TP99': 136.58, 'GPU Usage': '-'}}, {40: {'QPS': 499.79, 'TP50': 77.51, 'TP99': 138.69, 'GPU Usage': '-'}}]]


# r101 ours
```bash
img_name=hub.c.163.com/neteaseis/ai/torchpipe:0.4.2

nvidia-docker run -it --cpus=8  --name a108 --network=host --runtime=nvidia --privileged  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:/workspace   $img_name bash
cd examples/exp/ && pip install pynvml && export  CUDA_VISIBLE_DEVICES=1 

docker exec -it a108 bash
# resnet101_GPUPreprocess 
cd examples/exp/ && export CUDA_VISIBLE_DEVICES=3
# python decouple_eval/benchmark.py  --model resnet101  --preprocess gpu --preprocess-instances 3   --timeout 10 --max 8 --trt_instance_num 2 --total_number 20000 --client 30

python decouple_eval/benchmark.py  --model resnet101  --preprocess gpu --preprocess-instances 4  --max 8 --trt_instance_num 2  --timeout 2 --total_number 15000 --client 30

#{1: {'QPS': 323.13, 'TP50': 3.06, 'TP99': 4.12, 'GPU Usage': 66.0}}
#{5: {'QPS': 905.08, 'TP50': 5.51, 'TP99': 5.62, 'GPU Usage': 51.0}}
#{10: {'QPS': 1802.67, 'TP50': 5.43, 'TP99': 6.52, 'GPU Usage': 91.0}}
#{20: {'QPS': 2599.55, 'TP50': 7.67, 'TP99': 8.53, 'GPU Usage': 100.0}}
#{30: {'QPS': 2737.8, 'TP50': 10.82, 'TP99': 11.78, 'GPU Usage': 97.0}}
#{40: {'QPS': 2758.91, 'TP50': 14.31, 'TP99': 15.84, 'GPU Usage': 100.0}}


# resnet101_CPUPreprocess 
python decouple_eval/benchmark.py --model resnet101 --total_number 11000  --preprocess-instances 7 --max 8 --trt_instance_num 2 --timeout 2 --client 20
# timeout 5
# {1: {'QPS': 187.39, 'TP50': 5.26, 'TP99': 9.31, 'GPU Usage': 37.0}}
#  {5: {'QPS': 464.98, 'TP50': 10.73, 'TP99': 11.02, 'GPU Usage': 23.0}}
#  {10: {'QPS': 1099.94, 'TP50': 9.36, 'TP99': 11.13, 'GPU Usage': 50.0}}
# {20: {'QPS': 1915.0, 'TP50': 10.05, 'TP99': 23.47, 'GPU Usage': 72.0}}
# {30: {'QPS': 1903.41, 'TP50': 14.5, 'TP99': 31.39, 'GPU Usage': 70.5}}
# {40: {'QPS': 1909.47, 'TP50': 18.61, 'TP99': 35.12, 'GPU Usage': 70.0}}
# timeout 2
#{1: {'QPS': 189.24, 'TP50': 5.24, 'TP99': 6.33, 'GPU Usage': 37.0}}
#{5: {'QPS': 649.32, 'TP50': 7.69, 'TP99': 7.85, 'GPU Usage': 31.0}}
#{10: {'QPS': 1238.2, 'TP50': 8.05, 'TP99': 8.28, 'GPU Usage': 61.0}}
#{20: {'QPS': 1855.5, 'TP50': 9.58, 'TP99': 27.88, 'GPU Usage': 76.0}}
#{30: {'QPS': 1832.02, 'TP50': 13.95, 'TP99': 33.39, 'GPU Usage': 81.5}}
#{40: {'QPS': 1829.96, 'TP50': 19.24, 'TP99': 38.29, 'GPU Usage': 75.0}}
 

 <function run_gpu_preprocess_cmd at 0x7fbf04461040> [{1: {'QPS': 319.6, 'TP50': 3.08, 'TP99': 5.11, 'GPU Usage': 66.0}}, {5: {'QPS': 638.94, 'TP50': 8.87, 'TP99': 11.25, 'GPU Usage': 95.5}}, {10: {'QPS': 1784.36, 'TP50': 5.53, 'TP99': 6.48, 'GPU Usage': 92.0}}, {20: {'QPS': 2545.97, 'TP50': 7.83, 'TP99': 8.65, 'GPU Usage': 100.0}}, {30: {'QPS': 2700.27, 'TP50': 11.06, 'TP99': 11.66, 'GPU Usage': 99.5}}, {40: {'QPS': 2792.68, 'TP50': 14.14, 'TP99': 16.89, 'GPU Usage': 100.0}}]
<function run_cpu_preprocess_cmd at 0x7fbf044809d0> [{1: {'QPS': 188.16, 'TP50': 5.26, 'TP99': 7.29, 'GPU Usage': 37.0}}, {5: {'QPS': 648.85, 'TP50': 7.68, 'TP99': 7.88, 'GPU Usage': 31.0}}, {10: {'QPS': 1240.77, 'TP50': 8.02, 'TP99': 8.27, 'GPU Usage': 61.0}}, {20: {'QPS': 1820.54, 'TP50': 9.58, 'TP99': 32.25, 'GPU Usage': 76.5}}, {30: {'QPS': 1820.21, 'TP50': 13.91, 'TP99': 37.49, 'GPU Usage': 75.0}}, {40: {'QPS': 1808.5, 'TP50': 19.21, 'TP99': 44.56, 'GPU Usage': 78.0}}]



4 & 4
<function run_gpu_preprocess_cmd at 0x7f20c09d2040> [{1: {'QPS': 404.54, 'TP50': 2.42, 'TP99': 4.45, 'GPU Usage': 59.0}}, {5: {'QPS': 1464.12, 'TP50': 3.14, 'TP99': 5.16, 'GPU Usage': 78.0}}, {10: {'QPS': 2131.5, 'TP50': 4.47, 'TP99': 6.71, 'GPU Usage': 97.0}}, {20: {'QPS': 2394.2, 'TP50': 7.96, 'TP99': 14.98, 'GPU Usage': 98.0}}, {30: {'QPS': 2429.44, 'TP50': 12.14, 'TP99': 21.14, 'GPU Usage': 100.0}}, {40: {'QPS': 2381.95, 'TP50': 16.44, 'TP99': 34.99, 'GPU Usage': 96.0}}]
<function run_cpu_preprocess_cmd at 0x7f20c09f19d0> [{1: {'QPS': 170.25, 'TP50': 5.34, 'TP99': 10.48, 'GPU Usage': 35.0}}, {5: {'QPS': 518.99, 'TP50': 8.3, 'TP99': 19.57, 'GPU Usage': 40.5}}, {10: {'QPS': 1022.69, 'TP50': 8.38, 'TP99': 20.02, 'GPU Usage': 52.5}}, {20: {'QPS': 1486.91, 'TP50': 11.46, 'TP99': 37.62, 'GPU Usage': 68.0}}, {30: {'QPS': 1450.19, 'TP50': 16.18, 'TP99': 61.06, 'GPU Usage': 64.0}}, {40: {'QPS': 1350.98, 'TP50': 23.99, 'TP99': 83.68, 'GPU Usage': 40.0}}]
8 & 3
```
