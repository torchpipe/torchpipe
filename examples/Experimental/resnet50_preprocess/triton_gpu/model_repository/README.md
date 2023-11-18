triton cpu
https://github.com/triton-inference-server/python_backend/tree/main/examples/preprocessing


# triton gpu

```

python onnx_exporter.py  --save /tmp/model.onnx
mkdir -p ./model_repository/resnet50_trt/1
trtexec --onnx=/tmp/model.onnx --saveEngine=./resnet50_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 --fp16
```

```
tritonserver --model-repository=./model_repository
```