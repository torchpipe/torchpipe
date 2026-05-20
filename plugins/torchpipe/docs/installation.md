


## Quick Installation (PyPI)

```bash
img_name=nvcr.io/nvidia/pytorch:25.05-py3  # alternatives: 24.05, 23.05, 25.06, 24.03(for 1080, cudnn8)

docker run --rm --gpus all -it --network host \
    -v $(pwd):/pwd/ --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -w /pwd/ \
    $img_name \
    bash

pip install torchpipe
python -c "import torchpipe"
```

TorchPipe requires TensorRT headers and libraries in the system's linker search paths to enable TensorRT-related backends.  
If installed in non-standard locations, specify them via `TENSORRT_INCLUDE` and `TENSORRT_LIB`:

- `$TENSORRT_INCLUDE/NvInfer.h` must exist  
- `$TENSORRT_LIB/libnvinfer.so` must exist  
-  or you can `export FORCE_DOWNLOAD_TENSORRT=1`


## build Env Image yourself

You can build the base environment image as follows:

```bash
# GPU driver >= 550 required, cuda 12 compatible, support from 1080 Ti to 5090:
docker build -t torchpipe:base_trt93 -f docker/DockerfileCuda12_TRT93 .
```





## build from source



### Rebuild the core library Omniback
Omniback is usually not needed to be rebuilt.

However, if you want to modify the core library or encounter any compatibility issues, you can rebuild Omniback first.

```bash
git clone https://github.com/torchpipe/torchpipe.git --recursive
cd torchpipe/

curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env 

uv venv && source .venv/bin/activate



uv pip install --upgrade scikit_build_core fire ninja setuptools-scm setuptools apache-tvm-ffi 

export SETUPTOOLS_SCM_PRETEND_VERSION="..."

uv pip install -e . --no-build-isolation -v

python -c "import omniback; print(omniback.__version__)"

cd plugins/torchpipe

uv pip install -e . --no-build-isolation

python -c "import torchpipe"
```

## Dependency Compatibility


| Library |  Required Version | Recommended Version | Notes |
| :--- | :--- | :--- | :--- |
| **TensorRT** | [`8.5`, `~10.9`] | `9.3`, `10.9` | Not all version tested |
| **OpenCV** | `>=4` | `~=4.5.0` |  |
| **PyTorch** | `>=2.0` (0.2.0+), `>=1.13` (0.1.x) | `~=2.7.0` |  |
| **CUDA** |   [`11`,`12`] |  |  |

