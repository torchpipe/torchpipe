# Paddle2ONNX模型转化与预测

本章节介绍 PaddleOCR 模型如何转化为 ONNX 模型，并基于 ONNXRuntime 引擎预测。
修改自[paddle2onnx](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5/deploy/paddle2onnx), 添加了docker使用。

## 1. 环境准备

需要准备 PaddleOCR、Paddle2ONNX 模型转化环境，和 ONNXRuntime 预测环境

克隆PaddleOCR的仓库，使用release/2.4分支, 安装相关依赖

```bash
git clone  -b release/2.4 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

docker run --name ppocr -v $PWD:/paddle  --network=host -it paddlepaddle/paddle:2.2.2-gpu-cuda10.2-cudnn7 /bin/bash

pip install -r requirements.txt 
python3.7 -m pip install onnx onnxruntime==1.9.0  onnx-simplifier

python3.7 -m pip install paddle2onnx 
```
## 2. 模型转换


- Paddle 模型下载

有两种方式获取Paddle静态图模型：在 [model_list](../../doc/doc_ch/models_list.md) 中下载PaddleOCR提供的预测模型；
参考[模型导出说明](../../doc/doc_ch/inference.md#训练模型转inference模型)把训练好的权重转为 inference_model。

以 ppocr 中文检测、识别、分类模型为例：

```
wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar
cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && cd ..

wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar
cd ./inference && tar xf ch_PP-OCRv2_rec_infer.tar && cd ..

wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
cd ./inference && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar && cd ..
```

- 模型转换

使用 Paddle2ONNX 将Paddle静态图模型转换为ONNX模型格式：

```bash
paddle2onnx --model_dir ./inference/ch_PP-OCRv2_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/det_onnx/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/ch_PP-OCRv2_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/rec_onnx/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/ch_ppocr_mobile_v2.0_cls_infer \
--model_filename ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel \
--params_filename ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams \
--save_file ./inference/cls_onnx/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True


python -m onnxsim ./inference/det_onnx/model.onnx ./inference/det_sim.onnx 4 --dynamic-input-shape --input-shape 1,3,640,640

python -m onnxsim ./inference/rec_onnx/model.onnx ./inference/rec_sim.onnx  4 --dynamic-input-shape --input-shape 1,3,48,100

python -m onnxsim ./inference/cls_onnx/model.onnx ./inference/cls_sim.onnx 4 --dynamic-input-shape --input-shape 1,3,48,100
```
## 3. onnx推理

``` bash
python3.7 tools/infer/predict_system.py --use_gpu=False --use_onnx=True \
--det_model_dir=./inference/det_sim.onnx  \
--rec_model_dir=./inference/rec_sim.onnx  \
--cls_model_dir=./inference/cls_sim.onnx  \
--image_dir=./deploy/lite/imgs/lite_demo.png

# 以下保留所有检测结果
python3.7 tools/infer/predict_system.py --use_gpu=False --use_onnx=True \
--det_model_dir=./inference/det_sim.onnx  \
--rec_model_dir=./inference/rec_sim.onnx  \
--cls_model_dir=./inference/cls_sim.onnx  \
--image_dir=./deploy/lite/imgs/lite_demo.png --drop_score=0
```

---------------------------------------------------------------------------
