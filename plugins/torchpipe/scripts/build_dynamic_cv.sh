
# When implementing custom OpenCV-based backends, they must provide a compatible OpenCV development environment.

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


# If installed to a non-standard directory, 
# you must set the environment variables `OPENCV_INCLUDE` and `OPENCV_LIB` accordingly,
# ensuring that `$OPENCV_INCLUDE/opencv2/core.hpp` and `$OPENCV_LIB/libopencv_core.so` exist.