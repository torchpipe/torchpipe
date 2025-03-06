
- current

```toml
IoC[ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], TensorrtInferTensor; 
 CatSplit[S[GpuTensor,CatTensor],S[ContiguousTensor,TensorrtInferTensor,ProxyFromParam[post_processor]],SplitTensor]]


fix
constexpr auto DEFAULT_INIT_CONFIG =
  dag_entrypoint  "IoC[InstancesRegister[BackgroundThread[Reflect[backend]]], Register[Block[Batching, "
    "InstanceDispatcher]];Forward[node.{node_name}]]";
to
IoC[InstanceDispatcher,Aspect[Batching, "
    "InstanceDispatcher];Aspect],
node_entrypoint = IoC[InstancesRegister[BackgroundThread[BackendProxy]], Register[DI[Batching, InstanceDispatcher]]; Forward[node.{node_name}]]


InstanceDispatcher[*PreCat*, *CatedHandle*]
```

- python
反射 弱类型
二元逻辑而非多元逻辑

- 模型
black box
show by code 文档 后端

- wheel
manylinux_2_28 + manylinux2014
 



- HAMI_GENERATE_BACKEND   A[B]?    Z=A=>A;Z=A[B,C]=>C;Z=A[B,C[D]] => D;


[data ... data]
LoopCondition[time?instance?]
StateFul[Batching, InstanceDispatcher]
max 4 curr 3 timeout

[ ] Batching =》 init event (in dict_config)

<!-- 
Interpreter (forword parse)=> entrypoint  : DagDispatcher or 对应的节点

Interpreter (init parse)=> per-node initialization => 独立的dict_config
 List[InstancesRegister[BackgroundThread[BackendProxy]], "
    "Register[Aspect[Batching, "
    "InstanceDispatcher]]] -->

Interpreter:
```toml
entrypoint:

init=List[InstancesRegister[BackgroundThread[BackendProxy]], "
    "Register[Aspect[Batching, "
    "InstanceDispatcher]]]
```

string Dict Backend Event, List




<!-- 
-https://github.com/pytorch/tensordict/blob/main/GETTING_STARTED.md -->


support trt 9.3


- ThroughputBenchmark
https://github.com/pytorch/pytorch/blob/main/torch/utils/throughput_benchmark.py
https://github.com/pytorch/pytorch/blob/9b7130b8db62e9e550366419fa33c0f530d80beb/torch/csrc/utils/throughput_benchmark.cpp#L46


tensorboard timeline


Repost[Z, dddsa]

- entrypoint 从Interpreter独立


builtin types: Any, Dict, Backend, Event, Queue
builtin container: IoC Proxy/DI Sequential/S  Reflect[cls_name] Register/InstancesRegister InstanceFromParam Forward[instance_name]

CatSplit, DagDispatcher, InstanceDispatcher

Queue, Recv, Send  Q_i B_i B_c

BackgroundThread

二元操作而非多元操作，filter 

basic and simple parser rules：
 