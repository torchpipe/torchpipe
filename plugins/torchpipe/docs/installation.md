

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
# python -m pip install --upgrade pip # for 23.05, 22.12, 24.05
cd /workspace && pip install . && cd /workspace/plugins/torchpipe && pip install . --no-build-isolation


# JIT compile built-in backends
python -c "import torchpipe"
```




### Rebuild the core library Omniback: No isolation
Omniback is usually not needed to be rebuilt.

 However, if you want to modify the core library or encounter any compatibility issues, you can rebuild Omniback first.

```bash
git clone https://github.com/torchpipe/torchpipe.git --recursive
cd torchpipe/

python -m pip install --upgrade pip 

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

