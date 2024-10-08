# FROM nvcr.io/nvidia/cuda:11.6.0-devel-ubuntu20.04 as base_devel
# FROM nvcr.io/nvidia/cuda:11.6.0-runtime-ubuntu20.04 as base

# docker build --network=host -f ./docker/Dockerfile -t trt thirdparty/

FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04 as base_devel
FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 as base

# nvidia/ 前缀也可以


LABEL maintainer "tp-nan"
WORKDIR /app


# 1) base
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y vim ninja-build zip python3-pip wget && apt-get clean && apt-get autoclean && apt-get autoremove && rm -rf /root/.cache/*
RUN ln -s /usr/bin/python3 /usr/bin/python  && python -m pip install --upgrade pip 
RUN pip --no-cache-dir install torch numpy torchvision --index-url https://download.pytorch.org/whl/cu116 &&\
    rm -rf /root/.cache/

From base as thirdparty
# 2) TensorRT/opencv Installation
RUN wget https://codeload.github.com/opencv/opencv/zip/refs/tags/4.5.4 -O /app/opencv-4.5.4.zip

RUN unzip opencv-4.5.4.zip  && rm -rf /app/opencv-4.5.4.zip
RUN pip  --no-cache-dir install cmake==3.21
RUN cd opencv-4.5.4/ && sed -i '1a add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)' CMakeLists.txt  && mkdir build && cd build && \
        cmake -D CMAKE_BUILD_TYPE=Release \
            -D BUILD_WITH_DEBUG_INFO=OFF \
            -D CMAKE_INSTALL_PREFIX=/opencv_install \
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
                .. && make -j4 && make install


# Please manually download the TensorRT installation package from https://github.com/NVIDIA/TensorRT/tree/release/9.2.
# Here, we use tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-11.8.tar.gz as an example.
# Check if the file exists in the local directory
RUN if [ ! -f ./tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-11.8.tar.gz ]; then \
    # If not, download the file to the /app directory
    wget -P /app/ https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.3.0/tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-11.8.tar.gz; \
else \
    # If the file exists, copy it to the /app directory
    cp ./tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-11.8.tar.gz /app/; \
fi

RUN tar -xzvf /app/tensorrt-9.3.0.1.linux.x86_64-gnu.cuda-11.8.tar.gz -C /app/; 

RUN rm -rf /app/TensorRT-9.3.0.1/data && \
     rm -rf /app/TensorRT-9.3.0.1/targets/x86_64-linux-gnu/lib/*.a 

From  base_devel as base_devel_removed
# RUN rm /usr/local/cuda-11.6/targets/x86_64-linux/lib/*.a

RUN find /usr/local/cuda-11.6/targets/x86_64-linux/lib/ -type f \( -name 'lib*.a' ! -name 'libcudart_static.a' ! -name 'libcudadevrt.a' \) -delete

RUN rm /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusparse.so*
RUN rm /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusolverMg.so*
RUN rm /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcusolver.so*
RUN rm /usr/local/cuda-11.6/targets/x86_64-linux/lib/libcufft.so*

# 3) TensorRT opencv cuda Installation
From base
COPY --from=thirdparty /app/TensorRT-9.3.0.1/targets/x86_64-linux-gnu/lib/ /usr/lib/
COPY --from=thirdparty /app/TensorRT-9.3.0.1/include/ /usr/include/
COPY --from=thirdparty /opencv_install/ /usr/local/
COPY --from=thirdparty /app/TensorRT-9.3.0.1/python/tensorrt-*-cp38-none-linux_x86_64.whl /app/
COPY --from=thirdparty /app/TensorRT-9.3.0.1/bin/trtexec /usr/bin/

COPY --from=base_devel_removed  /usr/local/cuda-11.6/include/ /usr/local/cuda-11.6/include/
COPY --from=base_devel_removed /usr/local/cuda-11.6/bin/ /usr/local/cuda-11.6/bin
COPY --from=base_devel_removed /usr/local/cuda-11.6/nvvm /usr/local/cuda-11.6/nvvm
COPY --from=base_devel_removed /usr/local/cuda-11.6/targets/x86_64-linux/lib /usr/local/cuda-11.6/lib64
# COPY --from=base_devel_removed /usr/include/cudnn.h /usr/include/cudnn.h
# COPY --from=base_devel_removed /usr/lib/x86_64-linux-gnu/libcudnn.so* /usr/lib/x86_64-linux-gnu/
RUN pip install /app/tensorrt-*-cp38-none-linux_x86_64.whl && rm /app/tensorrt-*-cp38-none-linux_x86_64.whl

# git is needed for building PPL.CV
RUN apt-get update -y && apt-get install git -y && pip --no-cache-dir install cmake 
WORKDIR /app

# docker build --network=host -f ./docker/Dockerfile -t trt thirdparty/

# docker run --rm   --network=host --gpus=all  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -v `pwd`:/workspace --name="tmp" -it trt  /bin/bash 
# cd /workspace/ && WITH_CVCUDA=1 BUILD_PPLCV=1 python setup.py install 
# python -c "import torchpipe;print(torchpipe.__version__);print(torchpipe._C.get_sm())"
# # docker commit  tmp xxxx:latest


