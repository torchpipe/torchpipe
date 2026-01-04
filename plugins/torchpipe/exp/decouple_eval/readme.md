



```bash
# GPUPreprocess:
python decouple_eval/benchmark.py  --model empty --preprocess gpu --preprocess-instances 11 --total_number 40000

# {40: {'QPS': 5824.52, 'TP50': 4.94, 'TP99': 32.85, 'GPU Usage': 99.5}}
# 5242.068 =>  {10: {'QPS': 5541.77, 'TP50': 1.77, 'TP99': 2.05, 'GPU Usage': 95.0}}

python decouple_eval/benchmark.py  --model empty --preprocess gpu --preprocess-instances 11 --total_number 80000  --client 80
# a10
# {40: {'QPS': 10971.16, 'TP50': 3.63, 'TP99': 3.78, 'GPU Usage': 92.5}}
# 9873.9 => {11: {'QPS': 10408.18, 'TP50': 1.05, 'TP99': 1.09, 'GPU Usage': 88.0}}

# CPUPreprocess
python decouple_eval/benchmark.py  --model empty --preprocess cpu --preprocess-instances 24
# {40: {'QPS': 4569.43, 'TP50': 7.8, 'TP99': 21.48, 'GPU Usage': 12.0}}
# 4112.487 => {22: {'QPS': 4269.7, 'TP50': 5.63, 'TP99': 9.01, 'GPU Usage': 11.0}}

python decouple_eval/benchmark.py  --model empty --preprocess cpu --preprocess-instances 24 --total_number 40000 --client 80
# a10
# {80: {'QPS': 7375.19, 'TP50': 10.66, 'TP99': 12.28, 'GPU Usage': 11.0}}
# 6637.6 => {22: {'QPS': 6688.81, 'TP50': 3.25, 'TP99': 3.45, 'GPU Usage': 10.0}}


# resnet18_GPUPreprocess 
python decouple_eval/benchmark.py  --model resnet18 --preprocess gpu --preprocess-instances 11 --total_number 40000
# {40: {'QPS': 5324.25, 'TP50': 7.42, 'TP99': 12.15, 'GPU Usage': 100.0}}
# 4791.78 => {23: {'QPS': 5042.03, 'TP50': 4.55, 'TP99': 6.66, 'GPU Usage': 95.0}}
# a10
# {40: {'QPS': 9931.83, 'TP50': 4.01, 'TP99': 4.5, 'GPU Usage': 100.0}}
# 8937.9 => {30: {'QPS': 8959.87, 'TP50': 3.19, 'TP99': 4.44, 'GPU Usage': 100.0}}

# resnet18_CPUPreprocess 
python decouple_eval/benchmark.py --model resnet18 --preprocess-instances 24
# {40: {'QPS': 4551.78, 'TP50': 8.67, 'TP99': 13.22, 'GPU Usage': 41.0}}

python decouple_eval/benchmark.py --model resnet18 --preprocess-instances 24 --total_number 40000 --client 40
# a10
# {40: {'QPS': 7341.03, 'TP50': 5.37, 'TP99': 6.83, 'GPU Usage': 64.0}}
# 6606.9 => {32: {'QPS': 6659.89, 'TP50': 4.66, 'TP99': 6.25, 'GPU Usage': 62.0}}

python decouple_eval/benchmark.py --model resnet18 --preprocess-instances 30

# {40: {'QPS': 4880.7, 'TP50': 7.91, 'TP99': 13.41, 'GPU Usage': 41.0}}
```


```bash
# resnet101_GPUPreprocess 
python decouple_eval/benchmark.py  --model resnet101 --preprocess gpu --preprocess-instances 11 --total_number 10000
# {40: {'QPS': 3802.07, 'TP50': 10.32, 'TP99': 13.93, 'GPU Usage': 100.0}}
 
# resnet101_CPUPreprocess 
python decouple_eval/benchmark.py --model resnet101 --preprocess-instances 14 --max 4 --trt_instance_num 10
# {40: {'QPS': 3832.41, 'TP50': 10.18, 'TP99': 16.05, 'GPU Usage': 99.0}}
# a10:
# {1: {'QPS': 178.87, 'TP50': 5.1, 'TP99': 10.27, 'GPU Usage': 34.0}}
# {3: {'QPS': 287.35, 'TP50': 10.43, 'TP99': 10.58, 'GPU Usage': 20.0}}
# {5: {'QPS': 466.62, 'TP50': 10.7, 'TP99': 10.92, 'GPU Usage': 22.0}}
# {8: {'QPS': 1213.59, 'TP50': 6.52, 'TP99': 8.28, 'GPU Usage': 44.0}}
# {10: {'QPS': 1368.02, 'TP50': 6.6, 'TP99': 10.82, 'GPU Usage': 55.0}}
# {20: {'QPS': 2161.28, 'TP50': 8.98, 'TP99': 13.51, 'GPU Usage': 75.0}}
# {40: {'QPS': 2780.21, 'TP50': 14.98, 'TP99': 16.72, 'GPU Usage': 99.0}}
# {80: {'QPS': 2818.26, 'TP50': 28.2, 'TP99': 30.65, 'GPU Usage': 100.0}}
# {160: {'QPS': 2830.68, 'TP50': 56.32, 'TP99': 60.3, 'GPU Usage': 100.0}}
```


```bash
# faster_vit_1_224_GPUPreprocess 
python decouple_eval/benchmark.py  --model faster_vit_1_224 --preprocess gpu --preprocess-instances 11   --total_number 20000 --client 16 --timeout 15 
# {40: {'QPS': 2927.85, 'TP50': 13.37, 'TP99': 18.65, 'GPU Usage': 93.0}}
# 2635.065 => {30: {'QPS': 2673.07, 'TP50': 10.42, 'TP99': 21.5, 'GPU Usage': 84.0}}
# a10 {40: {'QPS': 2663.8, 'TP50': 14.74, 'TP99': 22.03, 'GPU Usage': 100.0}}
# 2397.42 => 

python decouple_eval/benchmark.py  --model faster_vit_1_224 --preprocess gpu --preprocess-instances 11  --max 4 --timeout 5 --trt_instance_num 10 --total_number 20000
# {40: {'QPS': 2336.63, 'TP50': 16.87, 'TP99': 23.11, 'GPU Usage': 100.0}}
# 2102.967 => {16: {'QPS': 2159.14, 'TP50': 7.41, 'TP99': 8.43, 'GPU Usage': 98.0}}

# faster_vit_1_224_CPUPreprocess 
python decouple_eval/benchmark.py --model faster_vit_1_224 --preprocess-instances 24
# {40: {'QPS': 3621.31, 'TP50': 10.75, 'TP99': 17.53, 'GPU Usage': 99.0}}


python decouple_eval/benchmark.py  --model faster_vit_1_224 --preprocess cpu --preprocess-instances 12  --max 16 --timeout 20 --trt_instance_num 3 --total_number 20000

```





# Predictability && under-batching
```bash
# 20250801

# docker
img_name=nvcr.io/nvidia/pytorch:25.05-py3
docker pull $img_name

docker run --name=zsy_omniback_all_cpu --runtime=nvidia --ipc=host  --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash
 
# test cuda
python -c  "import torch; assert (torch.cuda.is_available())"

# install omniback
python setup.py bdist_wheel
pip uninstall omniback -y && pip install dist/*.whl


# install torchpipe
cd plugins/torchpipe/
pip install -e .

# install timm
### optional: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install timm==1.0.15 onnxsim==0.4.36  py3nvml nvidia-pytriton==0.5.14 nvidia-ml-py -i https://pypi.tuna.tsinghua.edu.cn/simple

# to work directory
cd exp/


```

```bash
python decouple_eval/benchmark.py  --model resnet101 --preprocess cpu --preprocess-instances 14 --total_number 40000  --client 80
{1: {'QPS': 144.51, 'TP50': 5.29, 'TP99': 10.43, 'GPU Usage': 36.0}}
{10: {'QPS': 1312.82, 'TP50': 7.01, 'TP99': 10.6, 'GPU Usage': 57.0}}
{40: {'QPS': 2949.58, 'TP50': 13.36, 'TP99': 19.17, 'GPU Usage': 99.0}}
{80: {'QPS': 2913.99, 'TP50': 26.92, 'TP99': 38.13, 'GPU Usage': 100.0}}

python decouple_eval/benchmark.py  --model resnet101 --preprocess cpu --max 1 --trt_instance_num 40 --timeout 0 --preprocess-instances 14 --total_number 10000 --client 80
{1: {'QPS': 229.05, 'TP50': 4.35, 'TP99': 4.51, 'GPU Usage': 22.0}}
{10: {'QPS': 1297.24, 'TP50': 7.53, 'TP99': 10.5, 'GPU Usage': 98.0}}
{40: {'QPS': 1303.3, 'TP50': 30.27, 'TP99': 43.24, 'GPU Usage': 98.0}}
{80: {'QPS': 1244.96, 'TP50': 60.96, 'TP99': 207.66, 'GPU Usage': 97.0}}

taskset -c 12-40
python decouple_eval/benchmark.py  --model resnet101 --preprocess cpu --max 40 --trt_instance_num 1 --timeout 10 --preprocess-instances 14 --total_number 20000 --client 80
{1: {'QPS': 101.65, 'TP50': 7.57, 'TP99': 17.75, 'GPU Usage': 56.0}}
{10: {'QPS': 524.0, 'TP50': 19.06, 'TP99': 19.45, 'GPU Usage': 30.5}}
{40: {'QPS': 1840.08, 'TP50': 21.57, 'TP99': 23.46, 'GPU Usage': 54.0}}
{80: {'QPS': 3036.11, 'TP50': 25.97, 'TP99': 33.33, 'GPU Usage': 95.0}}

python decouple_eval/benchmark.py  --model resnet101 --preprocess cpu --max 4 --trt_instance_num 1 --timeout 10 --preprocess-instances 14 --total_number 20000 --client 80


```



```bash
export MODEL=resnet50


python decouple_eval/benchmark.py  --model $MODEL --preprocess cpu --max 1 --trt_instance_num 40 --timeout 0 --preprocess-instances 22 --total_number 10000 --client 80


python decouple_eval/benchmark.py  --model $MODEL --preprocess cpu --preprocess-instances 22 --total_number 40000  --client 80


python decouple_eval/benchmark.py  --model $MODEL --preprocess cpu --max 40 --trt_instance_num 1 --timeout 10 --preprocess-instances 22 --total_number 10000 --client 80


```

# yolox

```bash
python pipeline.py --config=pipeline_gpu.toml --benchmark --total_num 10000 --client 20 

python pipeline.py --config=pipeline_gpu.toml --benchmark --total_num 2000 --client 1

#
# {1: {'QPS': 32.77, 'TP50': 30.02, 'TP99': 40.14, 'GPU Usage': 13.0}}
# {5: {'QPS': 144.47, 'TP50': 35.42, 'TP99': 52.8, 'GPU Usage': 29.0}}
# {10: {'QPS': 292.44, 'TP50': 34.14, 'TP99': 51.07, 'GPU Usage': 43.5}}
#p90 {17: {'QPS': 711.93, 'TP50': 22.75, 'TP99': 33.97, 'GPU Usage': 85.5}}
# {20: {'QPS': 771.51, 'TP50': 26.41, 'TP99': 36.43, 'GPU Usage': 91.0}}
# {40: {'QPS': 783.41, 'TP50': 51.33, 'TP99': 62.36, 'GPU Usage': 91.0}}
# {80: {'QPS': 771.84, 'TP50': 101.67, 'TP99': 127.68, 'GPU Usage': 91.0}}

# a10
# {1: {'QPS': 46.79, 'TP50': 20.84, 'TP99': 31.57, 'GPU Usage': 20.0}}
# {5: {'QPS': 198.65, 'TP50': 24.8, 'TP99': 33.45, 'GPU Usage': 40.5}}
# {10: {'QPS': 458.96, 'TP50': 22.61, 'TP99': 33.09, 'GPU Usage': 76.0}}
# {12: {'QPS': 525.86, 'TP50': 23.04, 'TP99': 31.68, 'GPU Usage': 87.5}}
# {20: {'QPS': 581.34, 'TP50': 34.61, 'TP99': 44.42, 'GPU Usage': 91.0}}
# {40: {'QPS': 565.3, 'TP50': 70.59, 'TP99': 79.08, 'GPU Usage': 89.0}}
# {80: {'QPS': 570.22, 'TP50': 139.79, 'TP99': 151.64, 'GPU Usage': 91.0}}

```

```bash
python pipeline.py --benchmark
```



