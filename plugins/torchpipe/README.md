<div align="center">
<h1 align="center">TorchPipe</h1>
<h6 align="center">Ensemble Pipeline Serving With PyTorch Frontend</h6>  
</div>

TorchPipe is an alternative choice for Triton Inference Server, mainly featuring similar functionalities such as Shared-memory, Ensemble, and BLS mechanism. 

It is a multi-instance pipeline parallel library that acts as a bridge between lower-level acceleration libraries (such as TensorRT, OpenCV, CVCUDA) and RPC frameworks (e.g. Thrift). It offers a thread-safe function interface for the PyTorch frontend.

## Usage

Below are some usage examples, for more check out the [examples](./examples/).

### Initialize and Prepare Pipeline

```python
from torchpipe import pipe
import tempfile
import torch

with tempfile.TemporaryDirectory() as tmpdir:
    from torchvision.models.resnet import resnet18

    # create some regular pytorch model...
    model = resnet18(pretrained=True).eval().cuda()

    # create example model
    model_path = f"{tmpdir}/resnet18.onnx"
    x = torch.ones((1, 3, 224, 224)).cuda()
    torch.onnx.export(model, x, model_path, opset_version=17,
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    thread_safe_pipe = pipe{
        "preprocessor": {
            "backend": "S[DecodeTensor,ResizeTensor,CvtColorTensor,SyncTensor]",
            'instance_num': 2,
            'color':'rgb',
            'resize_h': '224',
            'resize_w': '224',
            'next': 'model',

        },
        "model": {
            "backend": "SyncTensor[TensorrtTensor]",
            "model": model_path,
            "model::cache": model_path.replace(".onnx", ".trt"),
            "max": '4',
            'batching_timeout': 4, # ms, timeout for batching
            'instance_num': 2,
            'mean': "123.675, 116.28, 103.53", 
            'std': "58.395, 57.120, 57.375", # merged into trt
        }
    }
```

### Execute

We can execute the returned ``thread_safe_pipe`` just like the original PyTorch model, but in a thread-safe manner.

```python
data = {'data': open('/path/to/img.jpg', 'rb').read()}
thread_safe_pipe(data) # <-- this is thread-safe
result = data['result']
```

## Setup

> Note: compiling torchpipe depends on the TensorRT c++ API. Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).  You may also try installing torchpipe inside one of the NGC PyTorch docker containers(e.g. nvcr.io/nvidia/pytorch:25.05-py3). Similar to the structure of Triton Inference Server, torchpipe is a plugin collection (TensorRT, OpenCV, etc...) built on top of the core library (named Hami, used to standardize both computation backends and scheduling backends).

### Install the torchpipe Python library

To install the torchpipe Python library, call the following

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

### Specific Installation Instructions
Compile with the torch cxx11_abi flag:
```bash
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```
If the output is True, then cxx11_abi is enabled.

Compile the core library Hami (root directory):
```bash
USE_CXX11_ABI=1 python setup.py install
```
Or install from PyPI (hami-core will be automatically installed when compiling torchpipe):

```bash
# cxx11_abi=False
bash -c 'tmpdir=$(mktemp -d) && pip download hami-core --platform manylinux2014_x86_64 --only-binary=:all: --dest $tmpdir --no-deps && pip install $tmpdir/hami_core-*.whl && rm -rf $tmpdir'
# cxx11_abi=True: 
pip install hami-core 

```
   


## How does it work?

## How to 


## Version Migration Notes 

The core functionality of TorchPipe (v0) has been extracted into the standalone Hami library.  


TorchPipe (v1, this version) is a collection of deep learning computation backend plugins built on the Hami library, primarily integrating third-party libraries including TensorRT, OpenCV, and LibTorch.