Yolo11: 
- 获取 n s m l, x 模型的 ONNX 格式

```bash

pip install --upgrade ultralytics


python yolo_onnx.py --model_size m --input_width -1 --input_height -1 --batch_size -1 --opset 18
python yolo_onnx.py --model_size l --input_width -1 --input_height -1 --batch_size -1 --opset 18


# python yolo_onnx.py --model_name yolo12 --model_size m --input_width -1 --input_height -1 --batch_size -1

# model_size can be n, s, m, l, x
```

- 运行模型

```bash
# 后处理在python中用pybind c++11加速,输出可视化结果. 正常有5个框
# python yolo_visual.py 
# 后处理用后端实现,输出可视化结果
# python yolo_visual_with_post.py 
# 用于实际高吞吐
python yolo_deploy.py --model yolo11m
```






