# Home


| Concurrent Requests[^2] | torch2trt | TorchPipe | TorchPipe w/ Thrift | Triton Inference Server |Triton Ensemble w/ DALI |
|:-------------------:|:---------:|:---------:|:-------------------:|:-----------------------:|:-----------------------:|
| 1                   | 90         | 124         | 92                   | 20   | 66    |
| 2                   | -         | 159         | 156                  | 45                     | 114       |
| 5                   | -         | 267        | 265                | 89                  | 233       |
| 10                  | -         | 315         | 304                   | 161                      | 307     |
| **Line of Code** | very low         | low         | low                   | middle                      | high     |




TorchPipe is a modular backend framework for DNN serving, designed on top of standardized computation and scheduling layers.  
Its primary goal is to deliver high performance with minimal user configuration.

If you find an issue, please [let us know](https://github.com/torchpipe/torchpipe/issues)!


[^2]: [Test environment](https://github.com/torchpipe/torchpipe/tree/main/plugins/torchpipe/examples/timm_model#deployment).
