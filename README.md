
[English](./README_en.md) | 简体中文




<div align="center">
<h1 align="center">torchpipe</h1>
<h6 align="center"><a href="https://pytorch.org/">Pytorch</a> 内的多线程流水线并行库</h6>

<!-- <img alt="license" src="https://img.shields.io/github/license/alibaba/async_simple?style=flat-square"> -->
<!-- <img alt="license" src="https://img.shields.io/github/license/alibaba/async_simple?style=flat-square">  -->
<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  -->
[![Documentation](https://img.shields.io/badge/torchpipe-Docs-brightgreen.svg)](https://torchpipe.github.io/zh/)
<!-- <img alt="language" src="https://img.shields.io/github/languages/top/torchpipe/torchpipe.github.io?style=flat-square"> -->
<!-- <img alt="feature" src="https://img.shields.io/badge/pytorch-Serving-orange?style=flat-square"> -->
<!-- <img alt="last commit" src="https://img.shields.io/github/last-commit/torchpipe/torchpipe.github.io?style=flat-square"> -->
</div>

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  -->





torchpipe是 介于底层加速库（如tensorrt，opencv，torchscript）以及 RPC（如thrift, gRPC）之间并与他们严格解耦的多实例流水线并行库；对外提供面向pytorch前端的线程安全函数接口，对内提供面向用户的细粒度后端扩展。



Torchpipe is a multi-instance pipeline parallel library that acts as a bridge between lower-level acceleration libraries (such as TensorRT, OpenCV, TorchScript) and RPC frameworks (like Thrift, gRPC), ensuring a strict decoupling from them. It offers a thread-safe function interface for the PyTorch frontend at a higher level, while empowering users with fine-grained backend extension capabilities at a lower level.





## 注意事项 
- 版本说明：推荐使用最新tag以及对应release
- main分支用于发布版本更新，develop分支用于提交代码和日常开发；


<!-- end elevator-pitch -->

## 快速开始

<!-- start quickstart -->


###  1. 安装 

参见 [安装文档](https://torchpipe.github.io/zh/docs/installation)


### 2. 获取恰当的模型文件(目前支持 onnx, trt engine等) 


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
 
### 3. 现在你可以并发调用单模型了

```python
import torch, torchpipe
model = torchpipe.pipe({'model': model_path,
                        'backend': "Sequential[cvtColorTensor,TensorrtTensor,SyncTensor]", # 后端引擎， 可见后端API参考文档。
                        'instance_num': 2, 'batching_timeout': '5', # 实例数和超时时间
                        'max': 4, # 模型优化范围最大值，也可以为 '4x3x224x224'
                        'mean': '123.675, 116.28, 103.53',#255*"0.485, 0.456, 0.406"，
                        'std': '58.395, 57.120, 57.375', # 将融合进tensorrt网络中
                        'color': 'rgb'}) # cvtColorTensor后端的参数： 目标颜色空间顺序
data = torch.zeros((1, 3, 224, 224)) # or torch.from_numpy(...)
input = {"data": data, 'color': 'bgr'}
model(input)  # 可多线程并行调用
# 使用 "result" 作为数据输出标识；当然，其他键值也可自定义写入
print(input["result"].shape)  # 失败则此键值一定不存在，即使输入时已经存在。
```

> 纯c++ API 可通过 [libtorch+cmake] 或者 [pybind11]的方式获得.

<!-- end quickstart -->
### 4. 我们的核心功能为多个节点间的一系列流水线设施。

> 更多信息，访问 [Torchpipe的文档](https://torchpipe.github.io/zh/docs/Inter-node) 。




### 5. RoadMap



torchpie目前处于一个快速迭代阶段，我们非常需要你的帮助。相对于贡献形式，我们对内容方面更为在意。欢迎通过issues或者merge requests等方式进行反馈。[贡献指南](https://torchpipe.github.io/zh/docs/contribution)。 

我们的最终目标是让服务端高吞吐部署尽可能简单。为了实现这一目标，我们将积极自我迭代，也愿意参与有相近目标的其他项目。

近期 RoadMap
- 大模型方面的示例
- 公开的基础镜像和pypi(manylinux)
- 优化编译系统，分为core,pplcv,model/tensorrt,opencv等模块
- 基础结构优化。包含python与c++交互，异常，日志系统，跨进程后端的优化；
- 技术报告

潜在未完成的研究方向

- 单节点调度和多节点调度后端，他们与计算后端无本质差异，需要更多面向用户进行解耦，我们想要将这部分优化为用户API的一部分；
- 针对多节点的调试工具。由于在多节点调度中，使用了模拟栈设计，比较容易设计节点级别的调试工具；
- 负载均衡

### 6. 致谢
我们的代码库使用或者修改后使用了多个开源库，请查看[致谢](./ACKNOWLEDGEMENTS.md)了解更多详细信息。


### 7. 交流提问
微信群： https://torchpipe.github.io/zh/docs/contribution_guide/communicate