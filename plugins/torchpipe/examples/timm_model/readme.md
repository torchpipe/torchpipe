



## Export timm model to onnx
```bash
pip install fire onnx_graphsurgeon "torch<2.9"
python export.py --model_name=eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --opset=20
# timm 好像有自带的export_onnx脚本
```

## Deployment

To accelerate inference, `torch2trt` can be used. For serving scenarios, **TorchPipe** or **Triton Inference Server** is recommended to support pipeline parallelism, adaptive batching, GPU preprocessing, and reduce head-of-line (HOL) blocking.

Benchmark results (throughput) with different concurrent requests are as follows (measured on 3080TI GPU, Xeon(R) Gold 6150 CPU @ 2.70GHz, 8 CPU cores):

| Concurrent Requests | torch2trt | TorchPipe | TorchPipe w/ Thrift | Triton Inference Server |Triton Ensemble w/ DALI |
|---------------------|-----------|-----------|---------------------|-------------------------|-------------|
| 1                   | 90         | 124         | 92                   | 20   | 66    |
| 2                   | -         | 159         | 156                  | 45                     | 114       |
| 5                   | -         | 267        | 265                | 89                  | 233       |
| 10                  | -         | 315         | 304                   | 161                      | 307     |
| **Line of Code** | very low         | low         | low                   | middle                      | high     |

- Note: TorchPipe and Triton Ensemble use GPU for preprocessing, while others use torchvision.transforms.
- Under a single concurrent request, TorchPipe dynamically adjusts the timeout based on the observed traffic pattern.



### Example Code
torchpipe installation and 

### 
```bash
pip install "torch<2.9" torchvision  thrift fire timm

 python benchmarks/bench.py --num_clients=1
  python benchmarks/bench.py --num_clients=2
   python benchmarks/bench.py --num_clients=5
    python benchmarks/bench.py --num_clients=10
```





