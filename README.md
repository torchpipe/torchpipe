<!-- <div align="center">
<h1 align="center">TorchPipe</h1> -->
<!-- <h6 align="center">Ensemble Pipeline Serving With PyTorch Frontend</h6>  
</div> -->

# Torchpipe

torchpipe is an alternative choice for Triton Inference Server, mainly featuring similar functionalities such as [Shared-momory](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html), [Ensemble](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models), and [BLS](https://github.com/triton-inference-server/python_backend#business-logic-scripting) mechanism.

 For serving scenarios, TorchPipe is designed to support multi-instance deployment, pipeline parallelism, adaptive batching, GPU-accelerated operators, and reduced head-of-line (HOL) blocking.It acts as a bridge between lower-level acceleration libraries (e.g., TensorRT, OpenCV, CVCUDA) and RPC frameworks (e.g., Thrift). At its core, it is an engine that enables programmable scheduling.

<a href="https://torchpipe.github.io/torchpipe/"><img src="https://img.shields.io/badge/-Documentation-brightgreen"/></a> <a href="https://torchpipe.github.io/torchpipe/"><img src="https://img.shields.io/badge/-Benchmark-brightgreen"/></a>



## update
- [20260104] We switched to [tvm_ffi](https://github.com/apache/tvm-ffi) to provide clearer C++-Python interaction.



## Usage

Below are some usage examples, for more check out the [examples](./plugins/torchpipe/examples/).

### Initialize and Prepare Pipeline

```python
from torchpipe import pipe
import torch

from torchvision.models.resnet import resnet101

# create some regular pytorch model...
model = resnet101(pretrained=True).eval().cuda()

# create example model
model_path = f"./resnet101.onnx"
x = torch.ones((1, 3, 224, 224)).cuda()
torch.onnx.export(model, x, model_path, opset_version=17,
                    input_names=['input'], output_names=['output'], 
                    dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

thread_safe_pipe = pipe({
    "preprocessor": {
        "backend": "S[DecodeTensor,ResizeTensor,CvtColorTensor,SyncTensor]",
        # "backend": "S[DecodeMat,ResizeMat,CvtColorMat,Mat2Tensor,SyncTensor]",
        'instance_num': 2,
        'color': 'rgb',
        'resize_h': '224',
        'resize_w': '224',
        'next': 'model',
    },
    "model": {
        "backend": "SyncTensor[TensorrtTensor]",
        "model": model_path,
        "model::cache": model_path.replace(".onnx", ".trt"),
        "max": '4',
        'batching_timeout': 4,  # ms, timeout for batching
        'instance_num': 2,
        'mean': "123.675, 116.28, 103.53",
        'std': "58.395, 57.120, 57.375",  # merged into trt
    }}
)
```

### Execute

We can execute the returned ``thread_safe_pipe`` just like the original PyTorch model, but in a thread-safe manner.

```python
data = {'data': open('/path/to/img.jpg', 'rb').read()}
thread_safe_pipe(data) # <-- this is thread-safe
result = data['result']
```

## Setup

> Note: compiling torchpipe depends on the TensorRT C++ API. Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).  You may also try installing torchpipe inside one of the NGC PyTorch docker containers(e.g. nvcr.io/nvidia/pytorch:25.05-py3).

### Installation
To install the torchpipe Python library, call the following

- [Inside NGC Docker Containers](./README.md#inside-ngc-docker-containers)
- [Quick Installation](./plugins/torchpipe/docs/installation.md#quick-installation)
- [uv environment](./plugins/torchpipe/docs/installation.md#uv-environment) 
- [Rebuild the core library Omniback](./plugins/torchpipe/docs/installation.md#rebuild-the-core-library-omniback).


#### Inside NGC Docker Containers
#### test on 25.05, 24.05, 23.05, and 22.12
```bash
git clone https://github.com/torchpipe/torchpipe.git
cd torchpipe/

img_name=nvcr.io/nvidia/pytorch:25.05-py3

docker run --rm --gpus all -it --rm --network host \
    -v $(pwd):/workspace/ --privileged \
    -w /workspace/ \
    $img_name \
    bash

# pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

cd /workspace/plugins/torchpipe && python setup.py install --cv2
```

> For [other NGC docker containers](./plugins/torchpipe/docs/installation.md#inside-ngc-docker-containers).

 


## How does it work?
See [Basic Usage](https://torchpipe.github.io/torchpipe/usage/basic_usage.html).

## How to add (or override) a backend

WIP

## Version Migration Notes 



TorchPipe (v1, this version)  is a collection of deep learning computation backends built on  Omniback library. Not all computation backends from TorchPipe (v0) have been ported to TorchPipe (v1) yet.
