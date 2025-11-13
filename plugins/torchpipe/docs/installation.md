# Installation

Follow these steps to get started using torchpipe.

<!-- ## Installation -->


### Quick Installation
```bash
git clone -b v1 https://github.com/torchpipe/torchpipe.git      
cd torchpipe/plugins/torchpipe

python setup.py install --cv2
# by default, torchpipe will check torch._C._GLIBCXX_USE_CXX11_ABI to set compilation options

# the '--cv2' enabled opencv-related backends support for whom needed.

# If you are not inside the NGC docker, you may need to download and build opencv first by running
# python download_and_build_opencv.py --install_dir ~/opencv_install
# export OPENCV_INCLUDE=~/opencv_install/include
# export OPENCV_LIB=~/opencv_install/lib

# TensorRT-related backends support is enabled by default, you may need to download and install tensorrt first by:
# python download_and_build_tensorrt.py --install_dir ~/tensorrt_install
# export TENSORRT_INCLUDE=~/tensorrt_install/include
# export TENSORRT_LIB=~/tensorrt_install/lib
```
 
### Inside NGC Docker(test on 25.05 and 22.12)
```bash
git clone -b v1 https://github.com/torchpipe/torchpipe.git
cd torchpipe/

img_name=nvcr.io/nvidia/pytorch:25.05-py3
# img_name=nvcr.io/nvidia/pytorch:22.12-py3 

docker run --rm --gpus all -it --rm --network host \
    -v $(pwd):/workspace/ --privileged \
    -w /workspace/ \
    $img_name \
    bash

cd /workspace/plugins/torchpipe && python setup.py install --cv2
```

### uv environment
```bash
git clone -b v1 https://github.com/torchpipe/torchpipe.git
cd torchpipe/plugins/torchpipe

python3 -m pip install uv
uv venv
source .venv/bin/activate # deactivate by 'deactivate' command if needed
uv pip install "torch>=2.7.1" omniback -i https://mirrors.aliyun.com/pypi/simple 
# For torch>=2.7.1, it is known torch._C._GLIBCXX_USE_CXX11_ABI==True by default. For pre-11 abi, install omniback by: 
# bash -c 'tmpdir=$(mktemp -d) && pip download omniback --platform manylinux2014_x86_64 --only-binary=:all: --dest $tmpdir --no-deps && uv pip install $tmpdir/omniback_core-*.whl && rm -rf $tmpdir'

python setup.py install --cv2
cd tests && pytest
```





### Rebuild the core library Omniback
Omniback is usually not needed to be rebuilt if you only use the precompiled torchpipe wheel by:

```bash
# python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" => False
bash -c 'tmpdir=$(mktemp -d) && pip download omniback --platform manylinux2014_x86_64 --only-binary=:all: --dest $tmpdir --no-deps && pip install $tmpdir/omniback_core-*.whl && rm -rf $tmpdir'
# python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" => True
pip install omniback 
```
 However, if you want to modify the core library or encounter any compatibility issues, you can rebuild Omniback first.

```bash
git clone -b v1 https://github.com/torchpipe/torchpipe.git --recurse-submodules
cd torchpipe/

USE_CXX11_ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))") python setup.py install

cd tests && pytest
```
