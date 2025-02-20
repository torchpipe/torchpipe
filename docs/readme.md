


[*] DecodeTensor
[*] DecodeMat ResizeMat CvtColorMat Mat2Tensor 
[] Tensor2Mat
[] TensorrtTensor
[] Onnx2Tensorrt


refactor core of torchpipe




Forward[]
Init[]



HasKey[Launch[node_name], Identity]

(has_key)A <==> HasKey[A,Identity]
A,(or)B <==> Or[A,B]
A,(swap)B <==> Swap[A,B]

bool judge(const dict& obj);

python -> 
```python
bool judge(obj: hami.dict): ...
register_judge(name="YourJudge",judge)
backend="Py::YourJudge[Forward[node_name], Identity]"
backend="Py::YourJudge[Forward[pipeline.node_name], Identity]"

```

Sequential 支持or and
Sequential[A,(or)S[B,C]]  => S[Or[A,S[B,C]]]
S[A,(or)B] => S[or[A,B]]
S[A,(has_key)B] <==> HasKey[B, Identity] 

S[A,B] == S[A,(swap)B] == Swap[A,B]

S[P1,(swap)P2,(swap)B] == Swap[Swap[P1,P2], B]
Sequential[And[A,B]]

SerialSkip[P] == BinaryConditional[Identity, Forward[pipeline.node_name]]
 BinarySelect[Identity, Forward[ring]]

 ConditionalSelect[] 
 ConditionalEntry[]
 register_select(name="YourSelection")
 <!-- register_entry(name="Or", filter_alias= "or")
 
 
 register_no_result_parser() -->
 S[DecodeTensor, (or)S[DecodeMat,Mat2Tensor]]

 Or[DecodeTensor, S[DecodeMat,Mat2Tensor]]

 Or
 Swap


S[A, (or)S[B, C], D]



register/register_single

#### gstreamer conditonal
https://stackoverflow.com/questions/64120365/gstreamer-pipeline-to-write-data-to-file-based-on-condition

https://gstreamer.freedesktop.org/documentation/tutorials/basic/dynamic-pipelines.html?gi-language=c



[a]
backend="S[Forward[pipeline.decode_gpu], (or)Forward[pipeline.decode_cpu]]"
backend="Or[Forward[pipeline.decode_gpu], Forward[pipeline.decode_cpu]]"

[decode_gpu]
<!-- backend="StreamGuard[DecodeTensor]" -->
backend="S[DecodeTensor,StreamGuard]"
next="decode_cpu"
[decode_cpu]
or=1
backend="S[DecodeMat,Mat2Tensor,StreamGuard]"


最终方案

```markdown
<!-- or=1 作为参数
Aspect[HasKey,B]

Aspect[BinaryCondition,B]
Aspect[HasNoKey,B]  <==> HasNoKey(key="restart")[B] -->



[x]=/ /,兼容
[x]or=1 作为参数(cornel case)
[x]HasKey(key=)[] 存在  *::dependency 则设置dependency
[x]NotHasKey(key=)[] 存在  *::dependency 则设置dependency
[x]Condition[]  存在 *::dependency 则设置dependency

[x]ThrowIfNoResult[A]
[x]ResultParser[A]

a= _C.init("Restart[Identity]")
PyResultParser[A]
 []按照  第一个字母顺序 初始化 or 配置init_order "a,b,c,d,c,e,d;a,d;3e" ";d" "dd;"

PyCondition
[x]Forward[A] 
[x]Init[A]
[x]Launch[A] 
[x]ioc_control

实例初始化： backend / thread instance （async2sync） / multiple thread ctl
=> 实例化 init= MultipleInstance[ThreadExector[{backend}]]
TASK_INDEX_KEY,  
ioc=>IOC容器,  MultipleInstance[ThreadExector[{backend}]]
scheduler=>：Batching / Choose/ {node_name}/index/{i}

ioc负责两个部分注册：  scheduler/instances

global entry: DAG  

entry => Aspect[Restart, Dag, Launch[{node_name}/Batching], Launch[{node_name}/Choose/{i}]] 

init="Restart, Dag"
init= "Batching,Choose, MultipleInstance[ThreadExector]"

:  DAG 初始化=》注册dag/{node_name}

 : (运行时dependency) Aspect[Restart, Dag Launch[{node_name}/MultipleInstance], Launch[{node_name}/ThreadExector/{i}]]
Aspect[Batching, Dispatch]
Aspect[MultipleInstance, ThreadExector, backend]

Aspect[Restart, DagDispatcher[Aspect[Batching, ExecuteDispatcher[Aspect[BackgroundThread, BackendProxy]]]]]


Aspect[Restart, DagDispatcher]



 


无result需要： 用map标记 或者 用or
 
 
forward=Aspect[Restart,Dag]

[node_name.0]
"InstanceDispatcher::dependency" = "Aspect[BackgroundThread, SyncBackendProxy]"
<!-- Aspect[Restart,Tree[DagDispatcher; Batching, InstanceDispatcher; BackgroundThread, BackendExector]] -->
逻辑节点[node_name]
<!-- backend=Launch[{node_name}/] -->
"IOC::dependency"="Forward[{node_name},{},{node_name}.2]"

final: 
节点级别初始化：   nodel-level IOC controler         三个位置调度 （before dag/ batching ）
-  per node  instances。0123 （instance_num）  Aspect[BackgroundThread, backend] 线程安全, 通过callback获取状态
- per node scheduler Aspect[Batching, InstanceDispatcher] 线程安全   
    (Batching)知道底层状态的方法：设置回调，传入状态结构体，通过callback修改状态； 
 
```toml


[] interperter -python
[]DagDispatcher


<!-- Aspect == Dispatcher -->

entry=Aspect[Restart,DagDispatcher]
```

Interperter:
- config parser (toml, etc.)
- per node [init] (default=)
- parse [entry]
- 
最小化扩展 by clearly list it

```

- main scheduler


# 三个位置调度 （before dag/ batching）
Aspect[Restart, DagDispatcher,Batching, InstanceDispatcher] 
Aspect[Restart, DagDispatcher;ContiguousBatching, InstanceDispatcher]
entry = Aspect[Restart,Dag]() => (Container, schedule = Aspect[Batching, InstanceDispatcher])


Aspect[Restart,Dag]
Dag::dependency

hami.dag(or schedule)// hami.instance("Restart")

Aspect[Launch[A] , D] 禁止代理改变dependency



configuration parser => IOC容器 / Dag 容器 
  

golbal_parser 进行单个层级的展开，也就是对于 A=B[C,D[E]] 展开为 {A=B, B::dependency=C,D[E]}
[optional] Dag配置加载 (node_name, next map)  透过反射，本身不持有依赖，而是通过反射向IOC容器获取对象实例 （可实现自己的）
[optional] 单个节点的配置启动问题。IOC容器查找scheduler配置，创建Backend，并注册为 pipeline.{node_name}
Restart[Dag容器]。 默认的scheduler为Aspect[Batching,MultiInstance,BackgroundThread，BackendCreator]， 依次注册为
 

with Restart as r:
    dag(data)

Aspect[HasKey,B,C, Launch[A], D] 
Aspect[HasKey,B,C, ProxyOfA, D]  允许代理改变dependency， 线程安全。A 需要是可forward实时改变dependency的。Proxy静态对象？

代码生成HAMI_PROXY_WITH_DEPENDENCY(Aspect, Restart, "EventGuard,RestartEvent");

 ProxyOfA=Proxy[A]

 minimized 元数据结构

```



 DAG :
 https://airflow.apache.org/docs/apache-airflow/1.10.1/concepts.
 



threadsafe queue / 进程间通信 (like ZeroMQ) / RPC framework (like thrift) 对外提供服务或者嵌入其他服务中



conner case in deep learning/thread local


典型的依赖类型 Sequential[A,B,C]    Select[A,B,C] Or[A,B]
以及他们的复合 A[B[C]]
复合形式的依赖类型 S[A[B,C[D]],E,S[F,G[H]]]

// A依赖于B
```c++
struct A{
    A(){
        dependency_ = new B();
    }
    void forward(int data){
        //...
        dependency_->forward(data);
        //...
    }
    B* dependency_{nullptr};
    // ...
};
```
```c++
struct A{
    void inject_dependency(B* dependency){dependency_=dependency;}
    void forward(int data, B* dependency){
        //...
        dependency->forward(data);
        //...
    }
    B* dependency_{nullptr};
    // ...
};
```