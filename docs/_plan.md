
- current

```toml
IoCV0[ModelLoadder[(.onnx)Onnx2Tensorrt,(.trt)LoadTensorrtEngine], TensorrtInferTensor; 
 CatSplit[S_v0[FixTensor,CatTensor],S_v0[ContiguousTensor,TensorrtInferTensor,ProxyFromParam[post_processor]],SplitTensor]]


fix
constexpr auto DEFAULT_INIT_CONFIG =
  dag_entrypoint  "IoCV0[InstancesRegister[BackgroundThread[Reflect[backend]]], Register[Block[Batching, "
    "InstanceDispatcher]];Forward[node.{node_name}]]";
to
IoCV0[InstanceDispatcher,Aspect[Batching, "
    "InstanceDispatcher];Aspect],
node_entrypoint = IoCV0[InstancesRegister[BackgroundThread[BackendProxy]], Register[DI_v0[Batching, InstanceDispatcher]]; Forward[node.{node_name}]]


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

[ ] Batching =》 init event (in kwargs)

<!-- 
Interpreter (forword parse)=> entrypoint  : DagDispatcher or 对应的节点

Interpreter (init parse)=> per-node initialization => 独立的kwargs
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
builtin container: IoCV0 Proxy/DI_v0 [SequentialV0/S  Reflect[key_to_cls_name, default=backend] Register/InstancesRegister InstanceFromParam Forward[instance_name]

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




NVI private: _init _forward _forward_with_dep



类型共识 https://blog.csdn.net/intel1985/article/details/122692791


Generator?

RandomSampleQueue -> default_src_queue  (上界)
RandomSampleQueue -> default_src_queue


conditional loop

Optional[Source] -> Loop[Process, Condition] -> Output

FileSource  -> max to queue(default_src_queue)
uniformSample -> max to queue(default_src_queue)

FileSource[create_name] 


[pathway](https://arxiv.org/pdf/2203.12533)

leixing gongshi: duogongshi

https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/pluginGuide.html

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/in_process.html

https://docs.google.com/presentation/d/1jzC_PZVXrVNSFVCW-V4cFXb6pn7zZ2CyP_Flwo05aqg/edit?pli=1#slide=id.g33a37e65d13_0_0
easy-to-hack 



vllm:
Next step 1: Support various scheduling policies
Priority-based scheduling
Fair scheduling
Predictive scheduling

Next step 2: Pluggable scheduler
E.g., workload-specific scheduler
E.g., different schedulers for different hardware backends



img_name=nvcr.io/nvidia/pytorch:25.02-py3
docker run --rm --gpus=all --ipc=host  --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash
docker run --gpus=all --ipc=host --name debug_hami --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

<!-- https://aijishu.com/a/1060000000503090 -->

https://torchpipe.github.io/docs/faq/remote-gdb


design:
ReqToTokenPool -> ReqToPageTable  [max_num_req x max_num_page_per_seq](w. page index)
 <!-- [max_num_req x 1] (w. last_page_len)  -->

TokenToKVPoolAllocator -> PageToKVPoolAllocator [manage free pages slots] [max_num_page x 1 KVCache]

KVCache -> k x v -> (max_num_page, head_num, head_dim) * 2 * num_layer


- leixinggongshi(key-value) leixingzhuanhuan  there is no one rule to get all
- inline scheduling instruc
<!-- ./parser -->

- objective-agnostic dispatch （aop dep.） dag restart cb 

- pipeline parallel everywhere

- ContinuousBatching 中心dispatcher
- relation with actor
- With