

English | [简体中文](README_zh.md)


<div align="center">
<h1 align="center">TorchPipe</h1>
A Minimalist High-Throughput Deep Learning Model Deployment Framework

Production-Grade：Within NetEase over billions of calls supported by Torchpipe everyday.
<h6 align="center">Ensemble Pipeline Serving for  <a href="https://pytorch.org/">Pytorch</a> Frontend</h6>


<!-- <img alt="license" src="https://img.shields.io/github/license/alibaba/async_simple?style=flat-square"> -->
[![Documentation](https://img.shields.io/badge/torchpipe-Docs-brightgreen.svg)](https://torchpipe.github.io)
<!-- <img alt="license" src="https://img.shields.io/github/license/alibaba/async_simple?style=flat-square">  -->
<!-- <img alt="language" src="https://img.shields.io/github/languages/top/torchpipe/torchpipe.github.io?style=flat-square"> -->
<!-- <img alt="feature" src="https://img.shields.io/badge/pytorch-Serving-orange?style=flat-square"> -->
<!-- <img alt="last commit" src="https://img.shields.io/github/last-commit/torchpipe/torchpipe.github.io?style=flat-square"> -->
</div>

## Installation

```bash
$ git submodule update --init --recursive
$ python setup.py install
```

<details>
    <summary>Other installation options</summary>

### Using NGC Docker Image
The easiest way is to choose NGC mirror for source code compilation (official mirror may still be able to run low version drivers through Forward Compatibility or Minor Version Compatibility).


- Minimum support `nvcr.io/nvidia/pytorch:21.07-py3` (Starting from 0.3.2rc3)
- Maximum support `nvcr.io/nvidia/pytorch:23.08-py3`


First, clone the code:
```bash
$ git clone https://github.com/torchpipe/torchpipe.git
$ cd torchpipe/ && git submodule update --init --recursive
```

Then start the container and if your machine supports [a higher version of the image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags), you can use the updated version of the Pytorch image.
```bash
img_name=nvcr.io/nvidia/pytorch:23.05-py3  # for tensort8.6.1, LayerNorm
# img_name=nvcr.io/nvidia/pytorch:22.12-py3  # For driver version lower than 510
docker run --rm --gpus=all --ipc=host  --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash
```

> NOTE: If you are using a transformer-ish model, it is strongly recommended to use TensorRT >= 8.6.1 (`nvcr.io/nvidia/pytorch:23.05-py3`) for supporting opset 17 for `LayerNormalization` and opset 18 `GroupNormalization`.

</details>



## Multi-Instance High-Throughput Serving
Torchpipe is a multi-instance pipeline parallel library that acts as a bridge between lower-level acceleration libraries (such as TensorRT, OpenCV, ppl.cv) and RPC frameworks (like Thrift, gRPC), ensuring a strict decoupling from them. It offers a thread-safe function interface for the PyTorch frontend at a higher level, while empowering users with fine-grained backend extension capabilities at a lower level.

<img alt="teaser" src="./docs/teaser.png">


## Minimalist Configuration

The configuration of AI pipelines can be a complex and error-prone task, often requiring deep technical expertise. TorchPipe is designed to be intuitive and user-friendly, allowing even those with limited technical background in AI deployment to configure and optimize their AI pipelines efficiently. This enables users to define complex pipeline behaviors without getting bogged down in intricate coding details.

The following example shows the configuration for detection using YOLO-X. By default, it supports cross-request batching and node-level pipeline parallelism, under a product-ready environment.

```toml
batching_timeout = 5  # Waiting timeout for cross-request batching 
precision = "fp16" 

[jpg_decoder]
backend = "Sequential[DecodeMat,cvtColorMat,ResizePadMat,Mat2Tensor,SyncTensor]"
color = "bgr"
instance_num = 5
max_h = 416
max_w = 416

# the next node to go after this node
next = "detect"

[detect]
backend = "Sequential[TensorrtTensor,PostProcYolox,SyncTensor]" 
batching_timeout = 5 # Waiting timeout for cross-request batching 
instance_num = 2 
max = 4  # maximum batchsize
model = "./yolox_tiny.onnx"
"model::cache" = "./yolox_tiny.trt"

# params for user-defined backend PostProcYolox:
net_h=416
net_w=416
```

<!-- ## Notes
-  Use the latest tag and corresponding release.
-  The main branch is used for releasing version updates, while the develop branch is used for code submission and daily development. -->

<!-- end elevator-pitch -->

## Quick Start

<!-- start quickstart -->

### 1. Get appropriate model file (currently supports ONNX, TensorRT engine, etc.).

```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True).eval().cuda()

import tempfile, os, torch
model_path =  os.path.join(tempfile.gettempdir(), "./resnet18.onnx") 
data_bchw = torch.rand((1, 3, 224, 224)).cuda()
print("export: ", model_path)
torch.onnx.export(resnet18, data_bchw, model_path,
                  opset_version=17,
                  do_constant_folding=True,
                  input_names=["in"], output_names=["out"],dynamic_axes={"in": {0: "x"},"out": {0: "x"}})

# os.system(f"onnxsim {model_path} {model_path}")
```
 
### 2. Now you can perform concurrent calls to a single model.

```python
import torch, torchpipe
model = torchpipe.pipe({'model': model_path,
                        'backend': "Sequential[cvtColorTensor,TensorrtTensor,SyncTensor]", # Backend engine, see backend API reference documentation
                        'instance_num': 2, 'batching_timeout': '5', # Number of instances and timeout time
                        'max': 4, # Maximum value of the model optimization range, which can also be '4x3x224x224'
                        'mean': '123.675, 116.28, 103.53', # 255*"0.485, 0.456, 0.406"
                        'std': '58.395, 57.120, 57.375', # Fusion into TensorRT network
                        'color': 'rgb'}) # Parameters for cvtColorTensor backend: target color space order
data = torch.zeros((1, 3, 224, 224)) # or torch.from_numpy(...)
input = {"data": data, 'color': 'bgr'}
model(input)  # Can be called in parallel with multiple threads
# Use "result" as the data output identifier; of course, other key values ​​can also be custom written
print(input["result"].shape)  # If failed, this key value must not exist, even if it already exists when input.
```

> c++ API is also possible through [libtorch+cmake] or [pybind11].


<!-- end quickstart -->

## Roadmap

TorchPipe is currently in a rapid iteration phase, and we greatly appreciate your help.  Feel free to provide feedback through issues or merge requests. Check out our [Contribution Guidelines](./CONTRIBUTING.md).

Our ultimate goal is to make high-throughput deployment on the server side as simple as possible. To achieve this, we actively iterate and are willing to collaborate with other projects with similar goals.

Current RoadMap:


- Optimization of the compilation system, divided into modules such as core, pplcv, model/tensorrt, opencv, etc.
- Optimization of the basic structure, including Python and C++ interaction, exception handling, logging system, compilation system, and cross-process backend optimization.
- Technical reports

Potential research directions that have not been completed:

- Single-node scheduling and multi-node scheduling backends, which have no essential difference from the computing backend, need to be decoupled more towards users. We want to optimize this part as part of the user API.
- Debugging tools for multi-node scheduling. Since stack simulation design is used in multi-node scheduling, it is relatively easy to design node-level debugging tools.
- Load balancing.

### Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](./ACKNOWLEDGEMENTS.md) for more details.

### 7. Questions & Discussions

- Wechat Group: https://torchpipe.github.io/zh/docs/contribution_guide/communicate

