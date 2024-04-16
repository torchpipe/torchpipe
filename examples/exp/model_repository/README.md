
# triton gpu

```

# /usr/src/tensorrt/bin/trtexec
onnx_path=./resnet101.onnx
/usr/src/tensorrt/bin/trtexec --onnx=$onnx_path --saveEngine=./model_repository/en/resnet_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 --fp16
```

```
export CUDA_VISIBLE_DEVICES=0
tritonserver --model-repository=./model_repository





img_name=nvcr.io/nvidia/tritonserver:23.06-py3
 
nvidia-docker run -it --rm --runtime=nvidia --privileged  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v `pwd`:/workspace   $img_name bash
 cd /workspace/
tritonserver --model-repository=/workspace/model_repository/en
```

```bash
python decouple_eval/benchmark.py  --model triton_resnet_ensemble --preprocess gpu --preprocess-instances 11   --total_number 20000 --client 16 --timeout 15 
```

 