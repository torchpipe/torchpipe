# resnet18 example
 


## Model Conversion
 

To convert the dynamic batch ONNX model, follow these steps:

```python
# you are in examples/resnet18/
import torch
import torchvision.models as models
resnet18 = models.resnet18().eval()
x = torch.randn(1,3,224,224)
onnx_save_path = "./resnet18.onnx"
torch.onnx.export(resnet18,
                  x,
                  onnx_save_path,
                  opset_version=17,
                  do_constant_folding=True,
                  input_names=["input"],            # 输入名
                  output_names=["output"],  # 输出名
                  dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}})
```


simplify this model:

```bash
    pip install onnx-simplifier
    onnxsim  ./resnet18.onnx ./resnet18.onnx 3 --test-input-shape=1,3,224,224
```


reference to [export to ONNX](https://torchpipe.github.io/docs/faq/onnx).

## Configuration
------------------

In the [resnet18.toml](./resnet18.toml) file, we specify the CPU pre-processing node and the model inference node.

> Using `next` to connect to the next one or more nodes is not necessary. For independent nodes, you can specify the `node_name` during forward propagation to enter the corresponding node.

## forward
 
Referring to `resnet18.py`, we can perform inference on a single image:

```python
import torch
import cv2
import os
import torchpipe as tp


if __name__ == "__main__":
    # prepare data:
    img_path = "../../test/assets/encode_jpeg/grace_hopper_517x606.jpg"
    img = cv2.imread(img_path, 1)
    img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]

    img = img.tobytes()

    toml_path = "./resnet18.toml"

    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY, TASK_INFO_KEY
    nodes = pipe(toml_path)

    def run(img):
        img_path, img = img[0]
        input = {TASK_DATA_KEY: img, "node_name": "jpg_decoder"}
        nodes(input) # 可多线程同时调用

        if TASK_RESULT_KEY not in input.keys():
            print("error : no result")
            return

        return  input[TASK_RESULT_KEY].cpu()

    run([(img_path, img)])
```

Meanwhile, we can also perform local processing with 10 threads:

```python
    
from torchpipe.utils.test import test_from_raw_file
test_from_raw_file(run, os.path.join("../..", "test/assets/encode_jpeg/"))
```

Achieving 926.31 qps with 10 threads of local processing:

```bash
    ------------------------------Summary------------------------------
    tool's version:: 0.10.0
    request client:: 10
    request batch::  1
    total number::   10000
    throughput::     qps:  926.31,   [qps:=total_number/total_time]
                    avg:  10.8 ms   [avg:=1000/qps*(request_batch*request_client)]
    latency::        TP50: 10.67   TP90: 14.11   TP99:  18.09 
                    MEAN: 10.75   -50,-40,-20,-10,-1: 19.84,20.53,23.29,26.62,46.03 ms
    cpu::            usage: 477.6%
    -------------------------------------------------------------------
```
## gpu decode
 



Running `python resnet18.py --config=resnet18_gpu_decode.toml` will enable GPU decoding. Typically, this will significantly improve throughput in CPU scenarios on the server side.


resnet18_gpu_decode.toml:

```toml

    # Schedule'parameter
    batching_timeout = 6 #默认的凑batch的超时时间
    instance_num = 3 

    [jpg_decoder]
    backend = " Sequential[DecodeTensor,ResizeTensor,cvtColorTensor, SyncTensor]  " 
    resize_h = 224
    resize_w = 224
    color = "rgb"

    next = "cpu_decoder"

    [cpu_decoder]
    filter="or"
    backend = "S[DecodeMat,ResizeMat,cvtColorMat,Mat2Tensor,SyncTensor]" 
    resize_h = 224
    resize_w = 224
    color = "rgb"

    next = "resnet18"

    [resnet18]
    backend = "SyncTensor[TensorrtTensor]" 

    min = "1x3x224x224" 
    max = "4x3x224x224" 
    # or max='4'
    model = "./resnet18.onnx" # or resnet18_merge_mean_std_by_onnx.onnx

    mean="123.675, 116.28, 103.53" # 255*"0.485, 0.456, 0.406"
    std="58.395, 57.120, 57.375" # 255*"0.229, 0.224, 0.225"

    instance_num = 2
```


Here, [`filter="or"`](https://torchpipe.github.io/docs/Intra-node/Sequential#filter_sequential) is used to fallback to CPU when GPU pre-processing fails. Since pre-processing failure is a rare event, this step is not necessary and can be skipped. If the pre-processing fails, the output result will not have the `TASK_RESULT_KEY`.