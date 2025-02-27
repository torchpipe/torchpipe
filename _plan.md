
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
node_entrypoint = IoC[RegisterInstances[BackgroundThread[BackendProxy]], RegisterNode[Aspect[Batching, InstanceDispatcher]]; Forward[node.{node_name}]]


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
 List[RegisterInstances[BackgroundThread[BackendProxy]], "
    "RegisterNode[Aspect[Batching, "
    "InstanceDispatcher]]] -->

Interpreter:
```toml
entrypoint:

MultipleNodesIoC[RegisterInstances[BackgroundThread[BackendProxy]], "
    "RegisterNode[Aspect[Batching, "
    "InstanceDispatcher]]; DagDispatcher]

IoC[RegisterInstances[BackgroundThread[BackendProxy]], "
    "RegisterNode[Aspect[Batching, "
    "InstanceDispatcher]]; Forward[node.{node_name}]]
```

string Dict Backend Event, List
