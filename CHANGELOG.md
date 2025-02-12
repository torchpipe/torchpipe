# Changelog

## v0.6.6 
- enhance llama2 example
- Add FakeInstance[x] backend (fake_instance_num settable)
- fix `Batching` to support non-contiguous-batching mode

## v0.6.5 (2024.10.25)
- Add force_layer_norm_pattern_fp32 (trt >= 10.5)
- Implemented partial streaming openai interface.
- Consolidate the code supporting TensorRT >= 8.5, 9, and 10.
- TensorRT <= 8.4 is deprecated now

## v0.6.3 (2024.09.27)
- Add lots of key-value base instruction backends
- non-root node can be used as the entry node now
- Support `S[A(key=value)[B]]` grammar
- fix the default max_workspace_size

## v0.6.2 (2024.08.14)
- docker/Dockerfilex updated to TensorRT 10.3
- input_reorder/output_reorder/batch_process is also supported for TensorRT 9 now. previously only TensorRT 10 is supported.

## v0.6.1 
- Support batching for already batched input. Set `scheduler = Batching` and ensure the size of each tensor is smaller than the max batch size. Set `request_size` or use `cal_request_size_method = "CalTorchBatchSize"` for torch-related tensors.
- `EnsureInputHasEvent[EventLoop]`: Add support for iterative scheduling with asynchronous data sources.
- Add full support for `torchpipe.Event`.
- Full support for `LlamaForCausalLM`.

## v0.6.0 (2024.07.19)
- fix docker/Dockerfilex
- torchpipe.utils.cpp_extension: Resolved the issue where matching the incorrect libipipe.so path resulted in  importing multiple libipipe.so files, causing problems with handling static variables.
- Forbidding the use of TorchAllocator in the default CUDA stream.
- `AppendPositionIDsTensor` to append a position ID tensor(with shape (1, seq_len)) to the input qkv tensor(s).
- Add support TensorRT 10.2 and add input_reorder/output_reorder
- Add a TensorRT plugin, TorchPlugin, to offload computation to another node, and use the PyTorch tensor as input and output.
- improved `Jump`



## v0.5.1 (2024.05.22)
- Add support for TensorRT 10.0
- remove all *at::* namespace, use *torch::* instead
- TensorRT 7 is deprecated now
- add TENSORRT_PATH to set the path of TensorRT
- torchpipe need pybind11 >= v2.7.0 now (for cvnp)
- add MultiModalEmbedsTensor for multimodal embedding

## v0.4.4 (2024.05.08)
- add c++2py for long long type;
- add PPLResizePadTensor; CpuTensor,NCHWTensor;
- add ProcessAdaptor for test_tools 
- add batch_process for TensorrtTensor
- BUG FIX: MapReduce and PipelineV3 may cause resource deadlock under specific extreme conditions.


## v0.4.3 (2024.02.27)
- Add suppport for tensorrt 9.3
- make docker image default to tensorrt 9.3
- Openvino: instance_num/multiple outputs/more tests
- Add CreateTensor

## v0.4.3 Beta 1 
- Simpler Docker environment
- Add  `torchpipe.libipipe.get_sm()`(or `torchpipe._C.get_sm()`) to get the GPU architecture


## v0.4.2 Beta 3 (2024.02.02)
- Fix build issues with OpenVINO 
- Add conversion from numpy to cv::Mat
- Implementation of OpenVINO backend 
- More clear error message when the input type for forward propagation does not meet requirements

## v0.4.2 Beta 1 
- [Compile] Make rebuild_if_exist default to False
- Mat2Tensor support data_format now

## v0.4.1 Beta 2 (2024.1.17)
- [FIX]: Older versions of TensorRT may fail to compile due to the absence of the 'deallocate' interface in IGpuAllocator.
- [FIX]: Compilation failure for old versions of PyTorch due to incompatible dlpack.

## v0.4.1 Beta 1
- CVCUDA can be used on Pascal GPUs now


## v0.4.0 (2024.1.12)
- Fixing parsing issues when there are extra  dots in the trt's name.(thx yx)
- Make IPIPE_KEY default avaliable
- improved gradio visualization

## v0.3.3 Beta 4 
- add an examples of yolov8 pipeline serving in [examples/pipeline](examples/pipeline)
- add register_backend register_filter
- add load_backend, load_filter
- add base dockerfile docker/trt9

- add gil lock for the Destructor of Python backend, to avoid the possible core dump when multiple Python backend instances are destroyed at the same time.

## v0.3.3 Beta 3 


- fix TorchScriptTensor for batching input; update docs for training-tool
- test_tools: Add output of median GPU usage rate
- fix cuda arch error for pytorch 1.13.1 on NVIDIA A10
- The backend does not allow the deletion of the node_name key, as it would cause the scheduling system to crash.
- Further support for the Python backend has been added.
- add quick concat for tensor with same storage
- Add force_range for TensorrtTensor, borrow_from, active_instances_grp for BaselineSchedule(experimental)

## v0.3.3 Beta 2 (2023.11.05)

- Fixed the issue where a value is still assigned to the `result` when an error occurs in calling pplcv
- Add the model::timingcache parameter to TensorrtTensor to speed up the construction of models with the same structure in llm
- Fix the error when the model input type and the actual input type are different (such as torch.int32 and torch.int64).
- Add conversion from tuple type to C++
- Add Range; Use dynamic programming to check for optimal match, for example, 10 <= [2,9]: 10 = 8 + 2

```
torchpipe.pipe(
        {"Interpreter::backend": "Range[S[TensorrtTensor,SyncTensor]]",
        "range":"8,19","max": 9,"min":2, "model": "a.onnx"})

assert 19 == self.model_trt.max()
assert 8 == self.model_trt.min()
```

## v0.3.3 Beta 1 (2023.10.31)
- Add support for tensorrt 9.1
- Remove incorrect shape checking when the model has multiple inputs with different shapes.
- TensorrtTensor does not support non-float input types; this version fixes this issue.
- finish open source code

## v0.3.2 Release Candidate 3 (2023.10.13)
 
- Fix the bug that compilation fails due to missing `setMaxThreads` when the TensorRT version is too low；Compiled successfully under the image nvcr.io/nvidia/pytorch:21.07-py3
-  Add ppl.cv compilation option (BUILD_PPLCV).

## v0.3.2 Release Candidate 2 (2023.09.19)
- torchpipe.utils.test.throughput: API updated. Dependency on `onnx` and `tensorrt` python libraries removed.
- timm: `python -m torchpipe.utils.test.throughput --model=resnet18 --config instance_num:2` and `test_throughput_from_timm`

## v0.3.2 Release Candidate (2023.09.05)

- Unit tests in test_engine.py may fail due to the use of random weights when creating resnet18. This issue has been fixed
- The dependency on the ONNX Python library introduced in previous version has been removed

## v0.3.2-beta2 (2023.08.31)

- filter: rename `Continue` to `Run`. Deprecate `Continue`.
- Torch: Supports cross-GPU device data transfer
- Added some helper functions:
    - torchpipe.utils.models.onnx_export
    - torchpipe.utils.test.throughput
    - torchpipe.utils.models.register_model
    - torchpipe.utils.models.list_models
    - torchpipe.utils.models.create_model
- Fixed a bug that caused infinite looping during initialization and crashed the program when using `SyncTensor[Sequential[…,SyncTensor]]`
- fix the issue of check_dynamic_batchsize failing for some Unsqueeze layers during initialization(0.3.0b4-0.3.2b1).

## v0.0.1
- start from 2021.09.28