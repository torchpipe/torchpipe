







## get pt file(our use your own trained model)
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## get onnx file
### method 1
```bash
pip install ultralytics onnx

yolo mode=export model=yolov8n.pt format=onnx dynamic=True simplify

onnxsim yolov8n.onnx yolov8n.onnx

# optinal check:
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.trt --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640 --shapes=images:4x3x640x640 --fp16 --warmUp=1000
```

### method 2 （recommended）
```bash
export onnx with fixed shape, which is a much simpler onnx file:

```bash
python export.py

onnxsim yolov8n.onnx yolov8n.onnx

# optinal check:
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.trt --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640 --shapes=images:4x3x640x640 --fp16 --warmUp=1000
```


## run
 ```
python yolo.py
```
