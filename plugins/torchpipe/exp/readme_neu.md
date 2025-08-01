
```bash
source /etc/network_turbo

git clone https://github.com/MachineLearningSystem/25Eurosys-NeuStream-AE.git
# a10:~/paper/NeuStream-AE
 vim SD_our_system.py
# modify line 160: device = "cuda:3" to  device = "cuda"
 
 vim Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/RTX4090_run_neustream_request500.sh
#   for rate_scale in "1.25" "2.5"  exit 0


 docker stop neu && docker rm neu
img_name=nvcr.io/nvidia/pytorch:25.05-py3
 docker run -it --runtime=nvidia  --ipc=host --network=host --privileged   --shm-size 1G  --ulimit memlock=-1   -v `pwd`:/mount   --name="neu"  --cap-add=SYS_PTRACE $img_name  bash



 cd /mount/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256
 

pip install diffusers==0.26.3 transformers==4.48.2 -i https://pypi.tuna.tsinghua.edu.cn/simple


vim /usr/local/lib/python3.12/dist-packages/diffusers/utils/dynamic_modules_utils.py
# from huggingface_hub import cached_download, hf_hub_download, model_info
from huggingface_hub import hf_hub_download, model_info

sh RTX4090_run_neustream_request500.sh


apt-get update
apt-get install git-lfs
git lfs install
git lfs pull
```