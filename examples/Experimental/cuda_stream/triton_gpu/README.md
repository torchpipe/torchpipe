


 
# triton gpu

```

python3 onnx_export.py  --save /tmp/model.onnx
mkdir -p ./model_repository/resnet50_trt/1
mkdir -p model_repository/ensemble_dali_resnet/1

# /usr/src/tensorrt/bin/trtexec
trtexec --onnx=/tmp/model.onnx --saveEngine=./model_repository/resnet_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 --fp16
```

```
export CUDA_VISIBLE_DEVICES=0
tritonserver --model-repository=./model_repository
```
