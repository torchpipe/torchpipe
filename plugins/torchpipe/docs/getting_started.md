# Getting Started

Follow these steps to get started using torchpipe.

## installation

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
uv pip install "torch>=2.7.1" hami-core -i https://mirrors.aliyun.com/pypi/simple 
# For torch>=2.7.1, it is known torch._C._GLIBCXX_USE_CXX11_ABI==True by default. For pre-11 abi, install hami-core by: 
# bash -c 'tmpdir=$(mktemp -d) && pip download hami-core --platform manylinux2014_x86_64 --only-binary=:all: --dest $tmpdir --no-deps && uv pip install $tmpdir/hami_core-*.whl && rm -rf $tmpdir'

python setup.py install --cv2
cd tests && pytest
```





## Rebuild the core library Hami
Hami is usually not needed to be rebuilt if you only use the precompiled torchpipe wheel by:

```bash
# python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" => False
bash -c 'tmpdir=$(mktemp -d) && pip download hami-core --platform manylinux2014_x86_64 --only-binary=:all: --dest $tmpdir --no-deps && pip install $tmpdir/hami_core-*.whl && rm -rf $tmpdir'
# python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)" => True
pip install hami-core 
```
 However, if you want to modify the core library or encounter any compatibility issues, you may need to rebuild Hami first.

```bash
git clone -b v1 https://github.com/torchpipe/torchpipe.git --recurse-submodules
cd torchpipe/

USE_CXX11_ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))") python setup.py install

cd tests && pytest
```
