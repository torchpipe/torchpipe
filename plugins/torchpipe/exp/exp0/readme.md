

# useless experiments
## multiple instances, with no preprocesser

resnet101, fp16, 32, 0.7, 4/5

```bash
 python benchmark.py --model=resnet18 --max 4 --trt_instance_num 8 --client=10
 # 1,1,2,4,8  ,16,32  
 # 1,2,2,2,1  ,2,1
  python benchmark.py --model=resnet101 --max 1 --trt_instance_num 1
    python benchmark.py --model=resnet101 --max 1 --trt_instance_num 2 
    python benchmark.py --model=resnet101 --max 2 --trt_instance_num 4 

    python benchmark.py --model=resnet101 --max 4 --trt_instance_num 2 
    trtexec --loadEngine=resnet101_b4_i1.trt --shapes=input:4x3x224x224
        python benchmark.py --model=resnet101 --max 4 --trt_instance_num 4 

        python benchmark.py --model=resnet50 --max 4 --trt_instance_num 4 
        python benchmark.py --model=resnet50 --max 4 --trt_instance_num 1 
    trtexec --loadEngine=resnet50_b4_i1.trt --shapes=input:4x3x224x224
     trtexec --loadEngine=resnet101_4.trt --shapes=input:4x3x224x224
   trtexec --loadEngine=resnet101_b4_i1.trt  --shapes=input:4x3x224x224

0.75 -> 2361.443

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b1.trt --shapes=input:1x3x224x224 --fp16 => 1029.22

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b2.trt --shapes=input:2x3x224x224 --fp16 => 1567.588

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b3.trt --shapes=input:3x3x224x224 --fp16 => 1938.114

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b4.trt --shapes=input:4x3x224x224 --fp16 => 2233.672

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b5.trt --shapes=input:5x3x224x224 --fp16 => 2453.145

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b6.trt --shapes=input:6x3x224x224 --fp16 => 2276.868

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b7.trt --shapes=input:7x3x224x224 --fp16 => 2417.093

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b8.trt --shapes=input:8x3x224x224 --fp16 => 2552.112

trtexec --onnx=resnet101.onnx --saveEngine==resnet101_b32.trt --shapes=input:32x3x224x224 --fp16 =>  3079.5104




  python benchmark.py --model=resnet101 --max 4 --trt_instance_num 8 
  python benchmark.py --model=resnet101 --max 8 --trt_instance_num 4 

 ```

## old

```bash
# GPUPreprocess:
python decouple_eval/benchmark.py  --model empty --preprocess gpu --preprocess-instances 11 --total_number 40000 --client 40

# {40: {'QPS': 5608.57, 'TP50': 7.01, 'TP99': 7.41, 'GPU Usage': 100.0}}
# 5047.713 =>  {10: {'QPS': 5571.9, 'TP50': 1.76, 'TP99': 2.22, 'GPU Usage': 93.5}}

# CPUPreprocess
python decouple_eval/benchmark.py  --model empty --preprocess cpu --preprocess-instances 24 --total_number 20000 --client 40
# {40: {'QPS': 5000.16, 'TP50': 7.34, 'TP99': 10.51, 'GPU Usage': 13.0}}
# 4500.144 => {24: {'QPS': 4667.12, 'TP50': 5.96, 'TP99': 8.91, 'GPU Usage': 12.0}}

# resnet18_GPUPreprocess 
python decouple_eval/benchmark.py  --model resnet18 --preprocess gpu --preprocess-instances 11 --total_number 40000 --client 40
# {40: {'QPS': 5230.34, 'TP50': 7.59, 'TP99': 9.27, 'GPU Usage': 100.0}}
# 4707.306 => {22: {'QPS': 4834.87, 'TP50': 4.62, 'TP99': 6.81, 'GPU Usage': 91.0}}

# resnet18_CPUPreprocess 
python decouple_eval/benchmark.py --model resnet18 --preprocess-instances 24 --client 40
# {40: {'QPS': 4753.39, 'TP50': 8.28, 'TP99': 11.75, 'GPU Usage': 42.0}}
# 4278.051 => {32: {'QPS': 4366.97, 'TP50': 7.25, 'TP99': 12.15, 'GPU Usage': 40.0}}



# {40: {'QPS': 4880.7, 'TP50': 7.91, 'TP99': 13.41, 'GPU Usage': 41.0}}

```



```bash
# resnet101_GPUPreprocess 
python decouple_eval/benchmark.py  --model resnet101 --preprocess gpu --preprocess-instances 11 --total_number 40000 --client 40
# {40: {'QPS': 3897.98, 'TP50': 10.06, 'TP99': 15.62, 'GPU Usage': 100.0}}
# 3508.18 => {33: {'QPS': 3567.18, 'TP50': 9.01, 'TP99': 12.42, 'GPU Usage': 100.0}}

# resnet101_CPUPreprocess 
python decouple_eval/benchmark.py --model resnet101 --preprocess-instances 24 --client 40
# {40: {'QPS': 3856.68, 'TP50': 10.16, 'TP99': 15.67, 'GPU Usage': 99.0}}
# 3471.013 => {36: {'QPS': 3543.59, 'TP50': 9.99, 'TP99': 16.13, 'GPU Usage': 94.0}}
```


```bash
# faster_vit_1_224_GPUPreprocess 
python decouple_eval/benchmark.py  --model faster_vit_1_224 --preprocess gpu --preprocess-instances 11   --total_number 20000
# {40: {'QPS': 2927.85, 'TP50': 13.37, 'TP99': 18.65, 'GPU Usage': 93.0}}
# 2635.065 => {30: {'QPS': 2673.07, 'TP50': 10.42, 'TP99': 21.5, 'GPU Usage': 84.0}}

# faster_vit_1_224_CPUPreprocess 
python decouple_eval/benchmark.py --model faster_vit_1_224 --preprocess-instances 24
# {40: {'QPS': 3621.31, 'TP50': 10.75, 'TP99': 17.53, 'GPU Usage': 99.0}}
```


```bash
# faster_vit_4_224_GPUPreprocess 
python decouple_eval/benchmark.py  --model faster_vit_4_224 --preprocess gpu --preprocess-instances 4  --max 16 --trt_instance_num 3 --timeout 15 --total_number 10000 
# {40: {'QPS': 735.68, 'TP50': 54.2, 'TP99': 61.16, 'GPU Usage': 98.0}}
# 662.112 => {29: {'QPS': 669.22, 'TP50': 43.32, 'TP99': 51.56, 'GPU Usage': 93.5}}

python decouple_eval/benchmark.py  --model faster_vit_4_224 --preprocess gpu --preprocess-instances 4  --max 8 --trt_instance_num 3 --timeout 15 --total_number 10000
# {40: {'QPS': 741.29, 'TP50': 54.67, 'TP99': 64.86, 'GPU Usage': 100.0}}
# 667.161 => {22: {'QPS': 674.16, 'TP50': 34.99, 'TP99': 37.41, 'GPU Usage': 100.0}}

python decouple_eval/benchmark.py  --model faster_vit_4_224 --preprocess gpu --preprocess-instances 4  --max 4 --trt_instance_num 2 --timeout 5 --total_number 10000 --client 40
#  {40: {'QPS': 631.19, 'TP50': 63.3, 'TP99': 64.42, 'GPU Usage': 100.0}}
# {10: {'QPS': 561.63, 'TP50': 17.82, 'TP99': 21.66, 'GPU Usage': 99.0}}
python decouple_eval/benchmark.py  --model faster_vit_4_224 --preprocess gpu --preprocess-instances 1  --max 1 --trt_instance_num 1 --timeout 0 --total_number 10000 --client 1
# {1: {'QPS': 133.64, 'TP50': 8.2, 'TP99': 8.45, 'GPU Usage': 50.0}}

# faster_vit_1_224_CPUPreprocess 
python decouple_eval/benchmark.py --model faster_vit_1_224 --preprocess-instances 24
#  
```