
[English](README.md) | 简体中文




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





torchpipe是 介于底层加速库（如tensorrt，opencv，CVCUDA, ppl.cv）以及 RPC（如thrift, gRPC）之间并与他们严格解耦的通用深度学习serving框架；对外提供面向pytorch前端的线程安全函数接口，对内提供面向用户的细粒度后端扩展。


torchpipe是 [Triton Inference Server](https://github.com/triton-inference-server/server) 的一个代替选择，主要功能类似于其[共享显存](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html)，[Ensemble](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models), [BLS](https://github.com/triton-inference-server/python_backend#business-logic-scripting)机制。

生产级别：在网易智企内部，每天有海量调用由Torchpipe支持。

## **注意事项**

- 建议选择以下两种方式测试多客户端同时发送请求下结果的一致性：
    - 少量输入（比如10张图片），在线校验每张图片输出结果相同
    - 大量输入（比如10000张图片），离线保存结果，校验多次一致性

tensorrt在max_batch_size=4时，很多时候输入1张和4张时结果有差异，这是正常的。但是此时固定输入只有有限种类（一般为2）结果

- *  由于缺乏人力，我们暂时不再维护单独的中文文档，建议查看[英文文档](./README.md)，或者加入torchpipe开源社区或者通过issue或者微信群提问即可*



### 6. 致谢
我们的代码库使用或者修改后使用了多个开源库，请查看[致谢](./ACKNOWLEDGEMENTS.md)了解更多详细信息。



### 相关链接
- [torchpipe: 知乎介绍](https://zhuanlan.zhihu.com/p/664095419)
- [RayServe](https://docs.ray.io/en/latest/serve/index.html)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [nndeploy](https://github.com/Alwaysssssss/nndeploy)
- [CV-CUDA](https://github.com/CVCUDA/CV-CUDA)



