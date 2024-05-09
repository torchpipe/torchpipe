```bash
# change number of client here
python resnet101.py --client=40
```

```python
{100: {'throughput::qps': 4251.59, 'latency::TP50': 23.32, 'latency::TP99': 33.99, 'cpu_usage': 2845.0, 'gpu_usage': 100.0}}
{90: {'throughput::qps': 4253.68, 'latency::TP50': 20.85, 'latency::TP99': 29.83, 'cpu_usage': 2814.2, 'gpu_usage': 100.0}}
{80: {'throughput::qps': 4264.15, 'latency::TP50': 18.61, 'latency::TP99': 25.42, 'cpu_usage': 2796.7, 'gpu_usage': 100.0}}
{70: {'throughput::qps': 4264.0, 'latency::TP50': 16.23, 'latency::TP99': 22.09, 'cpu_usage': 2756.5, 'gpu_usage': 99.0}}
{60: {'throughput::qps': 4260.16, 'latency::TP50': 13.93, 'latency::TP99': 19.1, 'cpu_usage': 2717.1, 'gpu_usage': 100.0}}
{50: {'throughput::qps': 4181.04, 'latency::TP50': 11.82, 'latency::TP99': 16.02, 'cpu_usage': 2593.2, 'gpu_usage': 100.0}}
{40: {'throughput::qps': 3829.49, 'latency::TP50': 10.22, 'latency::TP99': 15.33, 'cpu_usage': 2286.1, 'gpu_usage': 98.5}}
{30: {'throughput::qps': 2980.66, 'latency::TP50': 9.92, 'latency::TP99': 16.67, 'cpu_usage': 1838.9, 'gpu_usage': 83.0}}
{20: {'throughput::qps': 1551.95, 'latency::TP50': 13.48, 'latency::TP99': 20.09, 'cpu_usage': 1308.9, 'gpu_usage': 48.0}}
{10: {'throughput::qps': 578.52, 'latency::TP50': 17.77, 'latency::TP99': 24.43, 'cpu_usage': 694.9, 'gpu_usage': 31.5}}
{1: {'throughput::qps': 81.48, 'latency::TP50': 13.08, 'latency::TP99': 17.06, 'cpu_usage': 112.8, 'gpu_usage': 16.0}}

```
<!-- 
bs =1 
```bash
# change number of client here
python resnet101.py --config=./cfg/resnet101_bs1.toml --client=40
```

```python
{40: {'throughput::qps': 615.2, 'latency::TP50': 64.58, 'latency::TP99': 86.86, 'cpu_usage': 951.5, 'gpu_usage': 52.0}}
{30: {'throughput::qps': 624.14, 'latency::TP50': 47.75, 'latency::TP99': 63.56, 'cpu_usage': 968.0, 'gpu_usage': 53.0}}
{20: {'throughput::qps': 641.12, 'latency::TP50': 31.14, 'latency::TP99': 41.53, 'cpu_usage': 1004.5, 'gpu_usage': 53.0}}
{10: {'throughput::qps': 617.9, 'latency::TP50': 16.1, 'latency::TP99': 24.36, 'cpu_usage': 940.5, 'gpu_usage': 50.0}}
{5: {'throughput::qps': 374.14, 'latency::TP50': 13.16, 'latency::TP99': 19.52, 'cpu_usage': 594.2, 'gpu_usage': 36.0}}
{1: {'throughput::qps': 80.55, 'latency::TP50': 12.29, 'latency::TP99': 14.19, 'cpu_usage': 163.8, 'gpu_usage': 8.0}}
``` -->

 


bs =16*4 ()
```bash
# change number of client here
python resnet101.py --config=./cfg/resnet101_bs16.toml --client=40
```

```python
{60: {'throughput::qps': 3957.39, 'latency::TP50': 15.12, 'latency::TP99': 20.92, 'cpu_usage': 2586.0, 'gpu_usage': 99.0}}
{40: {'throughput::qps': 3048.67, 'latency::TP50': 12.82, 'latency::TP99': 20.49, 'cpu_usage': 2016.3, 'gpu_usage': 95.0}}
{10: {'throughput::qps': 630.37, 'latency::TP50': 15.79, 'latency::TP99': 23.09, 'cpu_usage': 946.7, 'gpu_usage': 76.0}}
{1: {'throughput::qps': 74.64, 'latency::TP50': 13.05, 'latency::TP99': 15.09, 'cpu_usage': 142.3, 'gpu_usage': 13.0}}
```

bs=40 (e41)
```bash
# change number of client here
python resnet101.py --config=./cfg/resnet101_bs40.toml --client=40
```

```python
{80: {'QPS': 3359.51, 'TP50': 24.36, 'TP99': 34.92, 'GPU Usage': 81.0}}
{40: {'QPS': 1331.35, 'TP50': 30.07, 'TP99': 36.72, 'GPU Usage': 34.5}}
{10: {'QPS': 379.75, 'TP50': 26.74, 'TP99': 28.45, 'GPU Usage': 23.0}}
{1: {'QPS': 65.67, 'TP50': 15.58, 'TP99': 25.88, 'GPU Usage': 30.5}}
```


bs=1  (e60) 
```bash
# change number of client here
python resnet101.py --config=./cfg/resnet101_bs1.toml --client=40
```

```python
{80: {'QPS': 620.49, 'TP50': 128.57, 'TP99': 158.02, 'GPU Usage': 53.0}}
{40: {'QPS': 609.91, 'TP50': 65.11, 'TP99': 88.2, 'GPU Usage': 52.0}}
{10: {'QPS': 611.52, 'TP50': 16.29, 'TP99': 24.73, 'GPU Usage': 50.0}}
{1: {'QPS': 80.09, 'TP50': 12.33, 'TP99': 14.58, 'GPU Usage': 8.0}}
```

bs=8,instance_num=5 (96.1)

```bash
# change number of client here
python resnet101.py --config=./cfg/resnet101.toml --client=40
```

```python
{80: {'QPS': 4259.08, 'TP50': 18.46, 'TP99': 27.03, 'GPU Usage': 99.0}}
{40: {'QPS': 3844.75, 'TP50': 10.14, 'TP99': 16.0, 'GPU Usage': 99.0}}
{10: {'QPS': 579.56, 'TP50': 17.79, 'TP99': 24.4, 'GPU Usage': 32.5}}
{1: {'QPS': 76.65, 'TP50': 13.19, 'TP99': 19.08, 'GPU Usage': 15.0}}
```


bs=2,instance_num=20 (96.1)

```bash
# change number of client here
python resnet101.py --config=./cfg/resnet101_bs2.toml --client=40
```

```python
{80: {'QPS': 1401.73, 'TP50': 56.78, 'TP99': 71.53, 'GPU Usage': 72.0}}
{40: {'QPS': 1459.43, 'TP50': 27.12, 'TP99': 37.79, 'GPU Usage': 76.0}}
{10: {'QPS': 781.29, 'TP50': 12.71, 'TP99': 19.12, 'GPU Usage': 41.0}}
{1: {'QPS': 82.89, 'TP50': 12.29, 'TP99': 17.53, 'GPU Usage': 9.0}}
```