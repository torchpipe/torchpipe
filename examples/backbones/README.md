
### 模块说明

该模块主要用于展示如何测试不同的backbone在torchpipe中的吞吐，为了方便您的测试，我们封装了torchpipe.utils.models.onnx_export接口以及torchpipe.utils.test.throughput接口，前者用于将需要测试的PyTorch模型转换为ONNX模型，以便torchpipe读取，后者用于吞吐的测试，如何使用这两个接口，可以参考test_throughput.py。

##### test throughput

同时我们提供了一个较为完备的测试吞吐的脚本test_throughput.py，脚本中集成了包括：resnet、convnextv2、fastervit等多个模型，只需运行即可得到测试结果，结果以markdown格式展示，您只需简单修改，便可以测试自己的模型。

##### register model

为了方便您进行批量的测试，我们设计了register_model模块，经过register的模型，只需要create_model([model_name])便可以的到模型，使您的代码更整洁，如果想要使用register功能可以参考register_model.py，register_model与test_throughput结合使用，会使得代码更简洁。


### 如何测试


#### 一、安装一些需要的库
```
pip install onnx onnxsim onnxruntime timm fastervit==0.9.4 pandas tabulate -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install polygraphy onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com # 可选，安装后，将对onnx做进一步的简化
```

#### 二、run test_throughput.py

如果跑我们预设的模型，直接运行即可

```
python test_throughput.py
```

如果想添加自己的模型，代码中修改直接加入自己的模型即可。



#### 三、测试结果

* 显卡: T4
* precision: fp16
* opset: 17
* input size: 1x3x224x224

|                 | resnet18   | resnet50   | convnextv2_atto   | convnextv2_femto   | convnextv2_pico   | convnextv2_nano   | convnextv2_tiny   | faster_vit_0_224   | faster_vit_1_224   |
|:----------------|:-----------|:-----------|:------------------|:-------------------|:------------------|:------------------|:------------------|:-------------------|:-------------------|
| tool's version  | 20230602.0 | 20230602.0 | 20230602.0        | 20230602.0         | 20230602.0        | 20230602.0        | 20230602.0        | 20230602.0         | 20230602.0         |
| num_clients     | 10         | 10         | 10                | 10                 | 10                | 10                | 10                | 10                 | 10                 |
| total_number    | 10000      | 10000      | 10000             | 10000              | 10000             | 10000             | 10000             | 10000              | 10000              |
| throughput::qps | 4201.44    | 1777.48    | 1759.42           | 1494.76            | 1072.79           | 728.01            | 316.17            | 1563.81            | 1012.22            |
| throughput::avg | 2.38       | 5.63       | 5.68              | 6.69               | 9.32              | 13.74             | 31.63             | 6.39               | 9.88               |
| latency::TP50   | 2.09       | 4.85       | 4.58              | 5.43               | 7.5               | 12.73             | 27.97             | 5.85               | 9.76               |
| latency::TP90   | 3.0        | 7.05       | 8.85              | 9.44               | 13.04             | 19.28             | 40.34             | 8.14               | 12.47              |
| latency::TP99   | 4.41       | 8.79       | 9.33              | 11.04              | 13.64             | 19.8              | 85.59             | 9.86               | 13.96              |
| latency::avg    | 2.37       | 5.62       | 5.67              | 6.68               | 9.31              | 13.72             | 31.61             | 6.38               | 9.87               |
| -50             | 6.69       | 10.08      | 13.07             | 12.38              | 16.25             | 22.24             | 405.99            | 14.72              | 14.46              |
| -20             | 6.76       | 10.53      | 13.31             | 15.63              | 18.17             | 32.05             | 454.47            | 15.48              | 17.3               |
| -10             | 6.84       | 12.21      | 16.11             | 20.27              | 20.53             | 33.13             | 462.14            | 15.71              | 23.6               |
| -1              | 7.04       | 15.77      | 25.3              | 25.12              | 25.8              | 36.95             | 471.02            | 15.96              | 37.83              |
| cpu_usage       | 0          | 205.9      | 204.9             | 204.9              | 204.9             | 203.5             | 155.7             | 205.2              | 204.0              |