#!/bin/bash
set -e  # 遇错即停

# ==================== 1. 环境准备 ====================
OPENCV_VERSION="4.12.0"
OPENCV_SRC_DIR="opencv-${OPENCV_VERSION}"
OPENCV_ZIP="opencv-${OPENCV_VERSION}.zip"

# ==================== 2. 下载并解压源码 ====================
if [ ! -d "$OPENCV_SRC_DIR" ]; then
    if [ ! -f "$OPENCV_ZIP" ]; then
        wget "https://codeload.github.com/opencv/opencv/zip/refs/tags/${OPENCV_VERSION}" -O "$OPENCV_ZIP"
    fi
    unzip -q "$OPENCV_ZIP" && rm -f "$OPENCV_ZIP"
fi

# apt install -y libopenblas-dev
# dnf install -y openblas-devel

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
# root=$(pwd)
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for abi_flag in 1 0; do
    cd $root_dir
    # 构建目录放在 OpenCV 源码目录内（推荐）
    BUILD_DIR="$root_dir/$OPENCV_SRC_DIR/build_${abi_flag}"
    INSTALL_PREFIX="/opencv_install/abi_flag${abi_flag}"

    echo ">>> 构建 ABI=${abi_flag} 版本（静态链接）..."

    # 清理旧构建
    rm -rf "$BUILD_DIR" "$INSTALL_PREFIX" 
    mkdir -p "$BUILD_DIR"

    # 进入构建目录（已在源码目录内）
    cd "$BUILD_DIR"

    #  -static-libstdc++ -static-libgcc
    # CMake 配置：关键静态链接 + 动态调度 + 最大兼容性
    cmake .. \
        -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_STANDARD_REQUIRED=ON \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
        -D CMAKE_INSTALL_LIBDIR=lib \
        -D CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${abi_flag} -march=x86-64 -mtune=generic" \
        -D BUILD_SHARED_LIBS=OFF \
        -D OPENCV_PYTHON_LINK_STATICALLY=OFF \
        -D BUILD_WITH_STATIC_CRT=ON \
        -D CV_ENABLE_INTRINSICS=ON \
        -D CPU_BASELINE=AVX2 \
        -D CPU_DISPATCH=ALL \
        -D ENABLE_FAST_MATH=ON \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_opencv_calib3d=OFF \
        -D BUILD_opencv_dnn=OFF \
        -D BUILD_opencv_features2d=OFF \
        -D BUILD_opencv_flann=OFF \
        -D BUILD_opencv_gapi=OFF \
        -D BUILD_opencv_highgui=OFF \
        -D BUILD_opencv_java=OFF \
        -D BUILD_opencv_ml=OFF \
        -D BUILD_opencv_objdetect=OFF \
        -D BUILD_opencv_photo=OFF \
        -D BUILD_opencv_stitching=OFF \
        -D BUILD_opencv_video=OFF \
        -D BUILD_opencv_videoio=OFF \
        -D BUILD_opencv_videostab=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_DOCS=OFF \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D WITH_MKL=OFF \
        -D WITH_JPEG=ON \
        -D BUILD_JPEG=ON \
        -D WITH_PNG=ON \
        -D BUILD_PNG=ON \
        -D BUILD_ZLIB=ON \
        -D WITH_TIFF=ON \
        -D BUILD_TIFF=ON \
        -D WITH_OPENEXR=OFF \
        -D WITH_WEBP=OFF \
        -D BUILD_opencv_python3=OFF \
        -D BUILD_opencv_python2=OFF \
        -D WITH_CUDA=OFF \
        -D WITH_CUDNN=OFF \
        -D WITH_V4L=OFF \
        -D WITH_QT=OFF \
        -D WITH_OPENGL=OFF \
        -D WITH_IPP=ON \
        -D BUILD_IPP=ON \
        -D BUILD_IPP_IW=ON  \
        -D IPP_IW_STATIC=ON \
        -D BUILD_opencv_gpu=OFF \
        -D WITH_ITT=OFF \
        -D WITH_BLAS=OFF \
        -D WITH_LAPACK=OFF \
        -D WITH_OPENBLAS=OFF 



        # -D WITH_IPP=ON \
        # -D BUILD_IPP_IW=ON \
        # -D IPP_IW_STATIC=ON \
        # -D OPENCV_DOWNLOAD_IPP=ON 

        # -D MKL_USE_STATIC_LIBS=ON \
        # -D MKL_USE_TBB=ON \

    # 编译安装
    make -j$(nproc) VERBOSE=1
    make install


    cd ../..
done




# ================== 1. 写入 C++ 源文件 ==================
cat > "$temp_dir/check_build.cpp" << 'EOF'
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << cv::getBuildInformation() << std::endl;
    return 0;
}
EOF

# ================== 2. 设置路径 ==================
rm -f $temp_dir/check_build

INSTALL_PREFIX="/opencv_install/abi_flag1"
# ================== 3. 编译 ==================
c++ -std=c++11 $temp_dir/check_build.cpp \
    -I"$INSTALL_PREFIX/include/opencv4" \
    -L"$INSTALL_PREFIX/lib" \
    -L$INSTALL_PREFIX/lib/opencv4/3rdparty/ \
    -Wl,-Bstatic \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_core \
        -ltbb -lipphal -lippiw -lippicv \
    -Wl,-Bdynamic \
    -ldl -lm -lpthread \
    -o $temp_dir/check_build

# ================== 4. 运行 ==================
chmod +x $temp_dir/check_build
$temp_dir/check_build

