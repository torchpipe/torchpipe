
- current

```toml
IoC[ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], TensorrtInferTensor; 
 CatSplit[S[GpuTensor,CatTensor],S[ContiguousTensor,TensorrtInferTensor,LaunchFromParam[post_processor]],SplitTensor]]


fix 
constexpr auto DEFAULT_INIT_CONFIG =
  dag_entrypoint  "IoC[RegisterInstances[BackgroundThread[Reflect[backend]]], RegisterNode[Block[Batching, "
    "InstanceDispatcher]];Forward[node.{node_name}]]";
to
IoC[InstanceDispitcher,Aspect[Batching, "
    "InstanceDispatcher];Aspect],
node_entrypoint = IoC[RegisterInstances[BackgroundThread[BackendProxy]], RegisterNode[Aspect[Batching, InstanceDispatcher]]; LaunchNode]

Launch[{post_processor}]


Fp16Tensor
Fp32Tensor
TypeTensor(0:fp16,1:fp32)
```

- python
反射 弱类型
二元逻辑而非多元逻辑

- 模型
black box
show by code 文档 后端

- wheel
manylinux_2_28 + manylinux2014
 