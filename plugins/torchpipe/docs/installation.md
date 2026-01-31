


## Quick Installation (PyPI)

```bash
img_name=nvcr.io/nvidia/pytorch:25.05-py3  # alternatives: 24.05, 23.05, 25.06, 24.04(for 1080)

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

If not found at runtime, TorchPipe will attempt to download them automatically.  

This is why using NGC Docker containers is recommended—they come with TensorRT and CUDA preconfigured (ensuring `torch.utils.cpp_extension.CUDA_HOME` is properly set).

## build Env Image yourself

You can build the base environment image as follows:

```bash
# GPU driver >= 550 required, cuda 12 compatible, support from 1080 Ti to 5090:
docker build -t torchpipe:base_trt93 -f docker/DockerfileCuda12_TRT93 .
```


### OpenCV JIT Environment Setup
When implementing custom OpenCV-based backends, they must provide a compatible OpenCV development environment.



```bash
wget https://codeload.github.com/opencv/opencv/zip/refs/tags/4.12.0 -O ./opencv-4.12.0.zip

unzip opencv-4.12.0.zip  && rm -rf ./opencv-4.12.0.zip
pip  --no-cache-dir install cmake

abiflag=$(python -c "import torch; print(int(torch.compiled_with_cxx11_abi()))")

cd opencv-4.12.0/ && mkdir build && cd build && \
        cmake -D CMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=$abiflag \
            -D BUILD_WITH_DEBUG_INFO=OFF \
            -D CMAKE_INSTALL_PREFIX=/usr/local/ \
            -D INSTALL_C_EXAMPLES=OFF \
            -D INSTALL_PYTHON_EXAMPLES=OFF \
            -DENABLE_NEON=OFF  \
            -D WITH_TBB=ON \
            -DBUILD_TBB=ON  \
            -DBUILD_WEBP=OFF \
            -D BUILD_ITT=OFF -D WITH_IPP=ON  \
            -D WITH_V4L=OFF \
            -D WITH_QT=OFF \
            -D WITH_OPENGL=OFF \
            -D BUILD_opencv_dnn=OFF \
            -DBUILD_opencv_java=OFF \
            -DBUILD_opencv_python2=OFF \
            -DBUILD_opencv_python3=ON \
            -D BUILD_NEW_PYTHON_SUPPORT=ON \
            -D BUILD_PYTHON_SUPPORT=ON \
            -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
            -DBUILD_opencv_java_bindings_generator=OFF \
            -DBUILD_opencv_python_bindings_generator=ON \
            -D BUILD_EXAMPLES=OFF \
            -D WITH_OPENEXR=OFF \
            -DWITH_JPEG=ON  \
            -DBUILD_JPEG=ON\
            -D BUILD_JPEG_TURBO_DISABLE=OFF \
            -D BUILD_DOCS=OFF \
            -D BUILD_PERF_TESTS=OFF \
            -D BUILD_TESTS=OFF \
            -D BUILD_opencv_apps=OFF \
            -D BUILD_opencv_calib3d=OFF \
            -D BUILD_opencv_contrib=OFF \
            -D BUILD_opencv_features2d=OFF \
            -D BUILD_opencv_flann=OFF \
            -DBUILD_opencv_gapi=OFF \
            -D WITH_CUDA=OFF \
            -D WITH_CUDNN=OFF \
            -D OPENCV_DNN_CUDA=OFF \
            -D ENABLE_FAST_MATH=1 \
            -D WITH_CUBLAS=0 \
            -D BUILD_opencv_gpu=OFF \
            -D BUILD_opencv_ml=OFF \
            -D BUILD_opencv_nonfree=OFF \
            -D BUILD_opencv_objdetect=OFF \
            -D BUILD_opencv_photo=OFF \
            -D BUILD_opencv_stitching=OFF \
            -D BUILD_opencv_superres=OFF \
            -D BUILD_opencv_ts=OFF \
            -D BUILD_opencv_video=OFF \
            -D BUILD_videoio_plugins=OFF \
            -D BUILD_opencv_videostab=OFF \
            -DBUILD_EXAMPLES=OFF \
            -DBUILD_opencv_calib3d=OFF \
            -DBUILD_opencv_features2d=OFF\
            -DBUILD_opencv_flann=OFF\
            -DBUILD_opencv_ml=OFF\
            -DBUILD_opencv_videoio=OFF\
                .. && make -j$(nproc) && make install
```

If installed to a non-standard directory, you must set the environment variables `OPENCV_INCLUDE` and `OPENCV_LIB` accordingly, ensuring that `$OPENCV_INCLUDE/opencv2/core.hpp` and `$OPENCV_LIB/libopencv_core.so` exist.



## build from source

### Inside NGC Docker Containers

#### test on 25.05, 24.05, 23.05, 25.06
```bash
git clone https://github.com/torchpipe/torchpipe.git
cd torchpipe/

img_name=nvcr.io/nvidia/pytorch:25.05-py3 # you can also try 24.05, 23.05, 25.06, but may need to upgrade pip: python -m pip install --upgrade pip

docker run --rm --gpus all -it --rm --network host \
    -v $(pwd):/workspace/ --ipc=host --ulimit memlock=-1 --ulimit stack=67108864\
    -w /workspace/ \
    $img_name \
    bash

# pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# python -m pip install --upgrade pip # for 23.05, 24.05
cd /workspace && pip install . -v && cd /workspace/plugins/torchpipe && pip install . --no-build-isolation


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

## Dependency Compatibility


| Library |  Required Version | Recommended Version | Notes |
| :--- | :--- | :--- | :--- |
| **TensorRT** | [`8.5`, `~10.9`] | `9.3`, `10.9` | Not all version tested |
| **OpenCV** | `>=4` | `~=4.5.0` |  |
| **PyTorch** | `>=1.13` | `~=2.7.0` |  |
| **CUDA** |   [`11`,`12`] |  |  |

