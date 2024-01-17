# Changelog

## v0.4.1 Beta 1 (WIP)
- CVCUDA can be used on Pascal GPUs now
- pack libcvcuda.so into the whl

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


