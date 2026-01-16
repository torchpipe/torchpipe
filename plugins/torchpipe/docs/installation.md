<!-- ## Installation -->
# Installation

pip install torchpipe
python -c "import torchpipe"


## build from source
### Inside NGC Docker Containers

#### test on 25.05, 24.05, 23.05, and 22.12
```bash
git clone https://github.com/torchpipe/torchpipe.git
cd torchpipe/

img_name=nvcr.io/nvidia/pytorch:25.05-py3 # you can also try 24.05, 23.05, 22.12, but may need to upgrade pip: python -m pip install --upgrade pip

docker run --rm --gpus all -it --rm --network host \
    -v $(pwd):/workspace/ --ipc=host --ulimit memlock=-1 --ulimit stack=67108864\
    -w /workspace/ \
    $img_name \
    bash

# pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# python -m pip install --upgrade pip # for 23.05, 22.12
cd /workspace && pip install . && cd /workspace/plugins/torchpipe && pip install . #  --no-deps --no-build-isolation 

# JIT compile built-in backends
python -c "import torchpipe"
```

#### test on 25.06
```bash
git clone https://github.com/torchpipe/torchpipe.git
cd torchpipe/

img_name=nvcr.io/nvidia/pytorch:25.06-py3

docker run --rm --gpus all -it --rm --network host \
    -v $(pwd):/workspace/ --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -w /workspace/ \
    $img_name \
    bash

cd /workspace/plugins/torchpipe 
python download_and_build_opencv.py
python setup.py install --cv2
```

### Quick Installation
```bash
pip install torchpipe



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

pip install .
cd tests && pytest
```





### Rebuild the core library Omniback: No isolation
Omniback is usually not needed to be rebuilt.

 However, if you want to modify the core library or encounter any compatibility issues, you can rebuild Omniback first.

```bash
git clone https://github.com/torchpipe/torchpipe.git --recursive
cd torchpipe/

python -m pip install --upgrade pip 
#  python -c "import torch; print(torch.__version__,torch.__file__, int(torch._C._GLIBCXX_USE_CXX11_ABI))"
pip install --upgrade scikit_build_core fire ninja setuptools-scm setuptools apache-tvm-ffi 

pip install . --no-deps --no-build-isolation -v

cd plugins/torchpipe

pip install . --no-deps --no-build-isolation -v 

python -c "import torchpipe"

```

### Dependency Compatibility


| Library |  Required Version | Recommended Version | Notes |
| :--- | :--- | :--- | :--- |
| **TensorRT** | [`8.5`, `~10.9`] | `9.3`, `10.9` | Not all version tested |
| **OpenCV** | `>=4` | `~=4.5.0` |  |
| **PyTorch** | `>=1.10.2` | `~=2.7.0` |  |
| **CUDA** |   [`11`,`12`] |  |  |

