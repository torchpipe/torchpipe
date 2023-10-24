

English | [简体中文](./README.md)

> Tensorrt`s native int8 ptq


## Step 1: Save the Input Image to the Network

```bash
mkdir ./cache_dir # Create a temporary directory to save the input

python save_img.py
```
The key is to save the network input in advance, which can be done using `torch.save("a.pt", f)`, or by using the [SaveTensor](https://torchpipe.github.io/docs/backend-reference/torch?_highlight=savetensor#savetensor) backend to save the network input at a suitable location. It is recommended to save no less than 500 different representative images.

## Step 2: Quantize the Images and Cache the Model


```bash
python int8.py
```

```bash
precision="int8" # or best
calibrate_input = "./cache_dir" # Specify the input data path
"model::cache" = "./resnet18.trt" # or ./resnet18.trt.encrypted
```

## Step 3: Use the cached .trt Model 

> The performance improvement of int8 is mainly reflected in the throughput improvement. It is recommended to test on Tesla T4 and deploy mainly on Tesla T4 (the int8 improvement ability of 1080Ti/2080Ti is much lower than that of Tesla T4), and pay attention to CPU bottlenecks.

> For detection models, it is recommended to skip the quantization of the last layer of output coordinates and scores (using the `precision::fp16` and `precision::fp32` parameters). Networks with regression layers typically require that the output tensors of these layers are not limited by the range of 8-bit quantization and may require higher representation accuracy than that provided by 8-bit quantization [*](https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/).

Throughput Testing:

```bash
python throughput.py  # This will load the model::cache
```

The inference QPS of the pure model of resnet18 on 1080TI has increased from 2147.09 to 3354.55.


## QAT
QAT Please refer to [./qat](./qat)。
