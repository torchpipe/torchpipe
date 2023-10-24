### Module Description

This module is mainly used to demonstrate how to test different backbones' throughput in torchpipe. To facilitate your testing, we have encapsulated the **torchpipe.utils.models.onnx_export** interface and the **torchpipe.utils.test.throughput** interface. The former is used to convert the PyTorch model to an ONNX model for torchpipe to read, and the latter is used for throughput testing. You can refer to **test_throughput.py** for how to use these two interfaces.

##### Test Throughput
At the same time, we provide a more complete script **test_throughput.py** for testing throughput. The script integrates multiple models such as resnet, convnextv2, fastervit, etc. You can get the test results by simply running it. The results are displayed in markdown format, and you can easily modify them to test your own models.

##### Register Model
In order to facilitate testing more models, we designed the register_model module. After registration, the model can be obtained by simply calling **create_model([model_name])**, making your code cleaner. If you want to use the register function, you can refer to register_model.py. The combination of register_model and test_throughput will make your code more concise.


### How to Test
1. Install some required libraries

```
pip install onnx onnxsim onnxruntime timm fastervit==0.9.4 pandas tabulate -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install polygraphy onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com # Optional, after installation, further simplification will be performed on ONNX
```

2. Run test_throughput.py

```
python test_throughput.py
```

If you want to add your own model, you can directly modify the code and add your own model to it.

#### 三、Result

* Nvidia GPU: T4
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