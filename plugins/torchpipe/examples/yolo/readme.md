Yolo11: 
- 获取 n s m l, x 模型的 ONNX 格式

```bash

pip install --upgrade ultralytics

# pip install opencv-python-headless~=4.5.0

python yolo_onnx.py --model_size m --input_width -1 --input_height -1 --batch_size -1 --opset 18
python yolo_onnx.py --model_size l --input_width -1 --input_height -1 --batch_size -1 --opset 18


# python yolo_onnx.py --model_name yolo12 --model_size m --input_width -1 --input_height -1 --batch_size -1

# model_size can be n, s, m, l, x
```

- 运行模型

```bash
# 后处理在python中用pybind c++11加速,输出可视化结果. 正常有5个框：
# python yolo_visual.py 

# 后处理用后端实现,输出可视化结果,并测试高吞吐 (实际部署时推荐使用)：
python yolo_deploy.py --model yolo11m
```








## FAQ:

- ImportError: libGL.so.1: cannot open shared object file: No such file or directory

```bash
apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
```