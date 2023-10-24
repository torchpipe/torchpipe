# 本部分示例为基于thrift通信完成resnet50的服务部署，模拟实际业务落地的情况。
## 环境安装
```bash
sh install.sh
```

## torchpipe 服务测试

```bash
# 需要提前安装thrift
# 服务端：
python server.py --torchpipe 1 --port 8888 &

# 测试qps：
python client_qps.py --port 8888 

# 模拟单个客户端：
python client.py --port 8888 
```

## torch2trt 服务测试
```bash
# 需要提前安装torch2trt
# 服务端：
python server.py --torchpipe 0 --port 8887 &

# 测试qps：
python client_qps.py --port 8887 

# 模拟单个客户端：
python client.py --port 8887 

```





## 速度测试（3080显卡）

为了达到性能较高的水平，我们分别给出了thrift20路和thrift10路的情况下的推荐配置:
 - 以thrift20路为例：  **前处理instance_num=8 , 模型max=4, instance_num=2/4**
 - 以thrift10路为例：  **前处理instance_num=4 , 模型max=4, instance_num=2**

- thrift20路的情况下，我们分别测试了前处理使用CPU版本和GPU版本的情况
 - 前处理CPU 

| thrift | 前处理 instance_num | 模型 instance_num | 模型max | QPS      |
|:------:|:-------------------:|:-----------------:|:-------:|:--------:|
| 20     | 8                   | 2                 | 4       | 2225.35  |
| **20**     | **8**                  | **4**                | **4**       | **2253.13**  |
| 20     | 4                   | 4                 | 4       | 1652     |
| 20     | 4                   | 2                 | 4       | 1706     |

 - 前处理GPU 

| thrift | 前处理 instance_num | 模型 instance_num | 模型max | QPS      |
|:------:|:-------------------:|:-----------------:|:-------:|:--------:|
| 20     | 8                   | 2                 | 4       | 2159.07 |
| **20**     | **8**                  | **4**                | **4**       | **2172.54**  |
| 20     | 4                   | 4                 | 4       | 1584.65    |
| 20     | 4                   | 2                 | 4       | 1589.57    |


- thrift10路的情况下，我们分别测试了前处理使用CPU版本和GPU版本的情况
 - 前处理CPU 

| thrift | 前处理 instance_num | 模型 instance_num | 模型max | QPS      |
|:------:|:-------------------:|:-----------------:|:-------:|:--------:|
| 10     | 8                   | 4                 | 4       | 1017     |
| 10     | 4                   | 4                 | 4       | 1555     |
| **10**     | **4**                   | **2**                 | **4**       | **1584**     |


- 前处理GPU 

| thrift | 前处理 instance_num | 模型 instance_num | 模型max | QPS      |
|:------:|:-------------------:|:-----------------:|:-------:|:--------:|
| 10     | 8                   | 4                 | 4       | 852.49     |
| 10     | 4                   | 4                 | 4       | 1348.43    |
| **10**     | **4**                   | **2**                 | **4**       | **1351.01**     |
