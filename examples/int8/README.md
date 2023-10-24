
[English](./README_en.md) | 简体中文

> Tensorrt`s native int8 ptq

## 第一步，保存网络输入图像
```bash
mkdir ./cache_dir # 创建临时目录保存输入
python save_img.py
```
其中关键是提前保存网络输入（可使用`torch.save("a.pt", f)`,
也可在适合位置使用 [SaveTensor](https://torchpipe.github.io/docs/backend-reference/torch?_highlight=savetensor#savetensor) 后端保存网络输入。建议保存不小于500张不同的有代表性的图片。

## 第二步，使用这些图像进行量化, 并缓存模型
```bash
python int8.py
```

```bash
precision="int8" # or best
calibrate_input = "./cache_dir" #指定输入数据路径
"model::cache" = "./resnet18.trt" # or ./resnet18.trt.encrypted
```

## 第三步，使用缓存的.trt模型即可

> 注意：int8的性能提升主要体现在吞吐提升上，建议在Tesla T4上进行测试，并且可以主要部署在Tesla T4上(1080Ti/2080Ti int8提升能力远小于Tesla T4)，并注意cpu瓶颈

> 对于检测模型，建议跳过输出坐标和分数的最后一层的量化。(通过参数 `precision::fp16`和`precision::fp32`). 具有回归层的网络通常要求这些层的输出张量不受8位量化的范围限制，并且可能需要比8位量化提供的精细度更高的表示精度[*](https://developer.nvidia.com/blog/improving-int8-accuracy-using-quantization-aware-training-and-tao-toolkit/)。
 

吞吐测试：
```bash
python throughput.py  # 将加载model::cache
```
resnet18在1080TI上纯粹模型的推理qps由2147.09增长到3354.55


## QAT
QAT 请参考[./qat](./qat)，以及[文档](https://torchpipe.github.io/docs/tools/quantization#qat)。

 