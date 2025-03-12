
- current

```toml
IoC[ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], TensorrtInferTensor; 
 CatSplit[S[FixTensor,CatTensor],S[ContiguousTensor,TensorrtInferTensor,ProxyFromParam[post_processor]],SplitTensor]]


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


 Limited Data types 
 basic: Bytes
 builtin types: Any, Dict, Backend
   (异步), Event, Queue(default queue), 
   （more schedule）TypedDict, KVStorage(req_id, resp_id, data) 
builtin container: IoC Proxy/DI Sequential/S  Reflect[key_to_cls_name, default=backend] Register/InstancesRegister InstanceFromParam Forward[instance_name]

CatSplit, DagDispatcher, InstanceDispatcher

Queue, Recv, Send Observer  Q_i B_i B_c

BackgroundThread

二元操作，filter 

basic and simple parser rules：
 

 test : pip install torch==2.7 --index-url https://download.pytorch.org/whl/nightly/cu118

 pip install torch==2.7 -i https://pypi.tuna.tsinghua.edu.cn/simple


 inline scheduling struc




```bash
pip3 install hami-core --platform manylinux2014_x86_64 --only-binary=:all: --target `python3 -c "import site; print(site.getusersitepackages())"`

# or system wide install: 
# pip3 install hami-core --platform manylinux2014_x86_64 --only-binary=:all:   --target `python3 -c "import site; print(site.getsitepackages()[0])"`

```



帮我实现一个类帮我实现一个KVStorage，c++17, 英语注释,类名自行修改。不要对外暴露InnerMap，也可以不要InnerMap。 如果insert时找不到req_id 抛异常，其他也类似，如get
要求：

- 双重unordered_map： KVStorage(req_id : str, key : str, value: std::any)
- 线程安全：内锁和外锁两把锁
- remove(req_id)
- optional<std::any> get
- insert(req_id, key, value) insert_or_assign
- ... 提供更多函数



NVI private: _init _forward _forward_with_dep



类型共识 https://blog.csdn.net/intel1985/article/details/122692791