## 生成trace
```

python generate_SD_trace.py
```

## 模型生成
```bash

git clone https://github.com/MachineLearningSystem/25Eurosys-NeuStream-AE.git
target_dir=./25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/
cp export_unetclipvaeonnx.py  $target_dir
cp modules.py  $target_dir

cd $target_dir

# repair:
# file_path: "/usr/local/lib/python3.10/dist-packages/diffusers/utils/dynamic_modules_utils.py", line 28
# from huggingface_hub import cached_download, hf_hub_download, model_info
=> from huggingface_hub import hf_hub_download, model_info
```

```bash
python export_unetclipvaeonnx.py

ls *.onnx -alh
cd -
```

## 生成profile信息
```bash
cd ../
ln -s ./exp4neustream/25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/model_parameters/    model_parameters
cd -
ln -s ./25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/stable_diffusion_v1_5 stable_diffusion_v1_5

mv ./25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/*.onnx .
mv ./25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/*.onnx.data .

mkdir tmp
USE_TRT=True python run_profile_trt.py --max_batch=48 
```


## run

```bash
bash run_omniback.sh

nohup bash run_omniback.sh > output.log 2>&1 &

```
-------

## 
cd ..
ln -s ./exp4neustream/25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/model_parameters/    model_parameters
cd -
ln -s ./25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/stable_diffusion_v1_5 stable_diffusion_v1_5
# file_path: "/usr/local/lib/python3.10/dist-packages/diffusers/utils/dynamic_modules_utils.py", line 28
# from huggingface_hub import cached_download, hf_hub_download, model_info
from huggingface_hub import hf_hub_download, model_info
```
```
#
cp export_unetclipvaeonnx.py  ./25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/
cp modules.py  ./25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/

cd ./25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/
python export_unetclipvaeonnx.py

ls 25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/*.onnx
25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/clip.onnx
25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/safety.onnx
25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/unet.onnx
25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/vae.onnx

# run neustream
sh run_neustream.sh
```