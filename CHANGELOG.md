# Changelog

## v0.3.3 Beta 2 (In development)

- Fixed the issue where a value is still assigned to the `result` when an error occurs in calling pplcv
- Add the model::timingcache parameter to TensorrtTensor to speed up the construction of models with the same structure in llm

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


