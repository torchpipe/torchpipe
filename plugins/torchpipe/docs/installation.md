<!-- ## Installation -->
# Installation



### Inside NGC Docker Containers

#### test on 25.05, 24.05, 23.05, and 22.12
```bash
git clone https://github.com/torchpipe/torchpipe.git
cd torchpipe/

img_name=nvcr.io/nvidia/pytorch:25.05-py3 # you can also try 24.05, 23.05, 22.12

docker run --rm --gpus all -it --rm --network host \
    -v $(pwd):/workspace/ --privileged \
    -w /workspace/ \
    $img_name \
    bash

# pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

cd /workspace/plugins/torchpipe && python setup.py install --cv2
```

#### test on 25.06
```bash
git clone https://github.com/torchpipe/torchpipe.git
cd torchpipe/

img_name=nvcr.io/nvidia/pytorch:25.06-py3

docker run --rm --gpus all -it --rm --network host \
    -v $(pwd):/workspace/ --privileged \
    -w /workspace/ \
    $img_name \
    bash

cd /workspace/plugins/torchpipe 
python download_and_build_opencv.py
python setup.py install --cv2
```

### Quick Installation
```bash
git clone https://github.com/torchpipe/torchpipe.git      
cd torchpipe/plugins/torchpipe

python setup.py install --cv2
# by default, torchpipe will check torch._C._GLIBCXX_USE_CXX11_ABI to set compilation options

# the '--cv2' enabled opencv-related backends support for whom needed.

# If you are not inside the NGC docker, you **may** need to download and build opencv first by running
# python download_and_build_opencv.py --install_dir ~/opencv_install
# export OPENCV_INCLUDE=~/opencv_install/include
# export OPENCV_LIB=~/opencv_install/lib

# TensorRT-related backends support is enabled by default, you **may** need to download and install tensorrt first by:
# python download_and_build_tensorrt.py --install_dir ~/tensorrt_install
# export TENSORRT_INCLUDE=~/tensorrt_install/include
# export TENSORRT_LIB=~/tensorrt_install/lib
```
 
 
### uv environment
```bash
git clone https://github.com/torchpipe/torchpipe.git
cd torchpipe/plugins/torchpipe

python3 -m pip install uv
uv venv # --python 3.11
source .venv/bin/activate # deactivate by 'deactivate' command if needed
uv pip install "torch" omniback

python setup.py install --cv2
cd tests && pytest
```





### Rebuild the core library Omniback
Omniback is usually not needed to be rebuilt.

 However, if you want to modify the core library or encounter any compatibility issues, you can rebuild Omniback first.

```bash
git clone https://github.com/torchpipe/torchpipe.git --recurse-submodules
cd torchpipe/

rm -rf dist/ && python -m build && pip install dist/*.whl

cd tests && pytest
```

### Dependency Compatibility


| Library |  Required Version | Recommended Version | Notes |
| :--- | :--- | :--- | :--- |
| **TensorRT** | [`8.5`, `~10.9`] | `9.3`, `10.9` | Not all version tested |
| **OpenCV** | - | `>=4.5.0` |  |
| **PyTorch** | - | `>=2.7.0` |  |
| **CUDA** |   [`11`,`12`] |  |  |

