

English | [简体中文](README.md)


<div align="center">
<h1 align="center">torchpipe</h1>
<h6 align="center">Accelerated <a href="https://pytorch.org/">Pytorch</a> Serving through Multithreading</h6>

<!-- <img alt="license" src="https://img.shields.io/github/license/alibaba/async_simple?style=flat-square"> -->
[![Documentation](https://img.shields.io/badge/torchpipe-Docs-brightgreen.svg)](https://torchpipe.github.io)
<!-- <img alt="license" src="https://img.shields.io/github/license/alibaba/async_simple?style=flat-square">  -->
<!-- <img alt="language" src="https://img.shields.io/github/languages/top/torchpipe/torchpipe.github.io?style=flat-square"> -->
<!-- <img alt="feature" src="https://img.shields.io/badge/pytorch-Serving-orange?style=flat-square"> -->
<!-- <img alt="last commit" src="https://img.shields.io/github/last-commit/torchpipe/torchpipe.github.io?style=flat-square"> -->
</div>


Torchpipe is a multi-instance pipeline parallel library that acts as a bridge between lower-level acceleration libraries (such as TensorRT, OpenCV, TorchScript) and RPC frameworks (like Thrift, gRPC), ensuring a strict decoupling from them. It offers a thread-safe function interface for the PyTorch frontend at a higher level, while empowering users with fine-grained backend extension capabilities at a lower level.







## Notes
-  Use the latest tag and corresponding release.
-  The master branch is used for releasing version updates, while the develop branch is used for code submission and daily development.



<!-- end elevator-pitch -->

## Quick Start

<!-- start quickstart -->


###  1. Installation

See [Installation](https://torchpipe.github.io/docs/installation).



### 2. Get appropriate model file (currently supports ONNX, TensorRT engine, etc.).



```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True).eval().cuda()

import tempfile, os, torch
model_path =  os.path.join(tempfile.gettempdir(), "./resnet18.onnx") 
resnet18 = models.resnet18(pretrained=True).eval().cuda()
data_bchw = torch.rand((1, 3, 224, 224)).cuda()
print("export: ", model_path)
torch.onnx.export(resnet18, data_bchw, model_path,
                  opset_version=17,
                  do_constant_folding=True,
                  input_names=["in"], output_names=["out"],dynamic_axes={"in": {0: "x"},"out": {0: "x"}})

# os.system(f"onnxsim {model_path} {model_path}")
```
 
### 3. Now you can perform concurrent calls to a single model.


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
### 4. Our core functionality is a series of pipeline facilities


> For more information, please visit the [Torchpipe documentation][quickstart-docs-en].

[quickstart-docs-en]: https://torchpipe.github.io/


### 5. Roadmap



torchpie is currently in a rapid iteration phase, and we greatly appreciate your help. We prioritize content over the contribution format. Feel free to provide feedback through issues or merge requests. Check out our [Contribution Guidelines](./CONTRIBUTING.md).

Our ultimate goal is to make high-throughput deployment on the server side as simple as possible. To achieve this, we actively iterate and are willing to collaborate with other projects with similar goals.

RoadMap for 2023 and 2024:

- Technical reports
- Examples of large models
- Optimization of the compilation system, divided into modules such as core, pplcv, model/tensorrt, opencv, etc.
- Optimization of the basic structure, including Python and C++ interaction, exception handling, logging system, compilation system, and cross-process backend optimization.

Potential research directions that have not been completed:

- Single-node scheduling and multi-node scheduling backends, which have no essential difference from the computing backend, need to be decoupled more towards users. We want to optimize this part as part of the user API.
- Debugging tools for multi-node scheduling. Since stack simulation design is used in multi-node scheduling, it is relatively easy to design node-level debugging tools.
- Load balancing.


### 6. Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](./ACKNOWLEDGEMENTS.md) for more details.

