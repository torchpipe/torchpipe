 
 




##  Thread-Safe Local Inference


For convenience, let's assume that the TensorRT inference functionality is encapsulated as a "computational backend" named `TensorrtTensor`. Since the computation occurs on the GPU device, we add `SyncTensor` to represent the stream synchronization operation on the GPU.

| Configuration | Parameter                    | Description                                                                                |
|---------------|------------------------------|--------------------------------------------------------------------------------------------|
| backend       | "SyncTensor[TensorrtTensor]" | The computational backend, like TensorRT inference itself, is not thread-safe.             |
| max           | 4                            | The maximum batch size supported by the model, used for model conversion (ONNX->TensorRT). |



By default, TorchPipe wraps an extensible single-node scheduling backend on top of this "computational backend," which provides the following three basic capabilities:

- Thread safety of the forward interface
- Multi-instance parallelism

    | Configuration | Default | Description                                                        |
    |---------------|---------|--------------------------------------------------------------------|
    | instance_num  | 1       | Perform inference tasks in parallel with multiple model instances. |


- Batching
 
    | Configuration    | Default | Description                                                                                                                        |
    |------------------|---------|------------------------------------------------------------------------------------------------------------------------------------|
    | batching_timeout | 0       | The timeout in milliseconds. |


### Performance tuning tips

Summarizing the above steps, we obtain the necessary parameters for inference of ResNet18 under TorchPipe:

 


```python
import torchpipe as tp
import torch

config = {
    # Single-node scheduler parameters: 
    "instance_num": 2,
    "batching_timeout": 5,
    # Computational backend:
    "backend": "SyncTensor[TensorrtTensor]",
    # Computational backend parameters:
    "model": "resnet18_-1x3x224x224.onnx",
    "max": 4
}

# Initialization
models = tp.pipe(config)
data = torch.ones(1, 3, 224, 224).cuda()

## Forward
input = {"data": data}
models(input) # <== Can be called from multiple threads
result: torch.Tensor = torch.from_dlpack(input["result"]) # "result" does not exist if the inference failed
```
 
 
 

Assuming that we want to support a maximum of 10 clients/concurrent requests, the `instance_num` is usually set to 2, so that we can handle up to `instance_num * max = 8` requests at most.

 
## Sequential

`Sequential` can link multiple backends together. In other words, `Sequential[DecodeTensor,ResizeTensor,CvtColorTensor,SyncTensor]` and `Sequential[DecodeMat,ResizeMat]` are valid backends.

During the forward execution of `Sequential[DecodeMat,ResizeMat]`, the data (dict) will go through the following process in sequence:

- Execute `DecodeMat`: `DecodeMat` reads `data` and assigns the result to `result` and `color`.
- Conditional control flow: attempts to assign the value of `result` in the data to `data` and deletes `result`.
- Execute `ResizeMat`: `ResizeMat` reads `data` and assigns the result to the `result` key.

`Sequential` can be abbreviated as `S`. 



## Custom backends

A major problem in business is that the preset backends (computational backend/scheduling backend/RPC backend/cross-process backend, etc.) cannot cover all requirements.   `Torchpipe` treat the backend itself is also an API oriented towards users. 
### Basic Types
#### any
Similar to `std::any` in C++17, we have defined a type-erased container, `omniback::any`, with an almost identical interface.
#### dict
As a data carrier, similar to Python's `dict`, we have also defined the following `dict` in C++:
```cpp
#ifndef CUSTOM_DICT
using dict = std::shared_ptr<std::unordered_map<std::string, omniback::any>>;
#else
#endif
```


### Backend
Torchpipe limits the basic elements of the backend to:

- Initialization: parameter configuration
- Forward: input/output interface
- max/min: batch range of data
