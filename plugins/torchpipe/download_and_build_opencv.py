import distutils.command.clean
import distutils.spawn
import glob
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
import importlib
import requests
import zipfile

POSSIBLE_OPENCV_LIB_DIR = set({"/usr/local/lib/"})
POSSIBLE_OPENCV_INCLUDE_DIR = set({"/usr/local/include/opencv4/"})

from pkg_resources import DistributionNotFound, get_distribution, parse_version
# from setuptools import find_packages, setup
import logging
def default_cxx11_abi():
    import torch
    return torch._C._GLIBCXX_USE_CXX11_ABI


def exist_return(install_dir):
        OPENCV_INCLUDE = os.path.join(install_dir, "include/opencv4")
        OPENCV_LIB = os.path.join(install_dir, "lib")
        opencv_found =  (Path(OPENCV_INCLUDE) / "opencv2/core.hpp").exists() and ( Path(OPENCV_LIB) / "libopencv_core.so").exists()
        if opencv_found:
            logging.warning(
                f" Opencv founded in {install_dir}.  Setting OPENCV_INCLUDE={OPENCV_INCLUDE} and OPENCV_LIB={OPENCV_LIB} -> SKIP the task.")
            return OPENCV_INCLUDE, OPENCV_LIB
        else:
            return None, None
        
def download_and_build_opencv(cxx11_abi : bool = default_cxx11_abi(), install_dir=None, force_reinstall=False):
    """
    Downloads and builds OpenCV from source if not already installed.
    """
    import tempfile
    tmp_dir = tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"build in tmp_dir: {tmp_dir}")

    if install_dir is None:
        install_dir = "/usr/local/"
    
    if not os.path.exists(install_dir):
        try:
            os.makedirs(install_dir, exist_ok=True)
        except:
            pass
        
    if not os.access(install_dir, os.W_OK):
        new_install_dir = os.path.expanduser("~/opencv_install")
        print(f"No write permission for {install_dir}. Using {new_install_dir} instead.")
        install_dir = new_install_dir

    OPENCV_INCLUDE, OPENCV_LIB = exist_return(install_dir)
    if not force_reinstall:
        if OPENCV_INCLUDE and OPENCV_LIB:
            return OPENCV_INCLUDE, OPENCV_LIB
        for dir_path in POSSIBLE_OPENCV_INCLUDE_DIR:
            if os.path.exists(os.path.join(dir_path, "opencv2/opencv.hpp")):
                OPENCV_INCLUDE = dir_path
                break
        for dir_path in POSSIBLE_OPENCV_LIB_DIR:
            if os.path.exists(os.path.join(dir_path, "libopencv_core.so")):
                OPENCV_LIB = dir_path
                break
        if OPENCV_INCLUDE and OPENCV_LIB:
            return OPENCV_INCLUDE, OPENCV_LIB
        
    OPENCV_VERSION = "4.5.4"
    OPENCV_URL = f"https://codeload.github.com/opencv/opencv/zip/refs/tags/{OPENCV_VERSION}"
    OPENCV_ZIP = f"opencv-{OPENCV_VERSION}.zip"
    OPENCV_DIR = os.path.join(tmp_dir, f"opencv-{OPENCV_VERSION}")

    
    os.chdir(tmp_dir)

    # Download OpenCV
    print(f"Downloading OpenCV {OPENCV_VERSION}...")
    response = requests.get(OPENCV_URL, stream=True)
    with open(OPENCV_ZIP, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract OpenCV
    print(f"Extracting {OPENCV_ZIP}...")
    with zipfile.ZipFile(OPENCV_ZIP, "r") as zip_ref:
        zip_ref.extractall()

    # Install CMake if not already installed
    print("Installing CMake...")
    subprocess.run(["pip3", "install", "cmake"], check=True)

    # Build OpenCV
    print(f"Building OpenCV {OPENCV_VERSION}...")
    
    os.chdir(OPENCV_DIR)
    print(f"build in {OPENCV_DIR}")
    with open("CMakeLists.txt", "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"add_definitions(-D_GLIBCXX_USE_CXX11_ABI={int(cxx11_abi)})\n" + content)

    os.makedirs("build", exist_ok=True)
    os.chdir("build")
    subprocess.run([
        "cmake",
        "-D", "CMAKE_BUILD_TYPE=Release",
        "-D", "BUILD_WITH_DEBUG_INFO=OFF",
        "-D", f"CMAKE_INSTALL_PREFIX={install_dir}",
        "-D", "INSTALL_C_EXAMPLES=OFF",
        "-D", "INSTALL_PYTHON_EXAMPLES=OFF",
        "-D", "ENABLE_NEON=OFF",
        "-D", "BUILD_WEBP=OFF",
        "-D", "BUILD_ITT=OFF",
        "-D", "WITH_V4L=OFF",
        "-D", "WITH_QT=OFF",
        "-D", "WITH_OPENGL=OFF",
        "-D", "BUILD_opencv_dnn=OFF",
        "-D", "BUILD_opencv_java=OFF",
        "-D", "BUILD_opencv_python2=OFF",
        "-D", "BUILD_opencv_python3=ON",
        "-D", "BUILD_NEW_PYTHON_SUPPORT=ON",
        "-D", "BUILD_PYTHON_SUPPORT=ON",
        "-D", "PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3",
        "-D", "BUILD_opencv_java_bindings_generator=OFF",
        "-D", "BUILD_opencv_python_bindings_generator=ON",
        "-D", "BUILD_EXAMPLES=OFF",
        "-D", "WITH_OPENEXR=OFF",
        "-D", "WITH_JPEG=ON",
        "-D", "BUILD_JPEG=ON",
        "-D", "BUILD_JPEG_TURBO_DISABLE=OFF",
        "-D", "BUILD_DOCS=OFF",
        "-D", "BUILD_PERF_TESTS=OFF",
        "-D", "BUILD_TESTS=OFF",
        "-D", "BUILD_opencv_apps=OFF",
        "-D", "BUILD_opencv_calib3d=OFF",
        "-D", "BUILD_opencv_contrib=OFF",
        "-D", "BUILD_opencv_features2d=OFF",
        "-D", "BUILD_opencv_flann=OFF",
        "-D", "BUILD_opencv_gapi=OFF",
        "-D", "WITH_CUDA=OFF",
        "-D", "WITH_CUDNN=OFF",
        "-D", "OPENCV_DNN_CUDA=OFF",
        "-D", "ENABLE_FAST_MATH=1",
        "-D", "WITH_CUBLAS=0",
        "-D", "BUILD_opencv_gpu=OFF",
        "-D", "BUILD_opencv_ml=OFF",
        "-D", "BUILD_opencv_nonfree=OFF",
        "-D", "BUILD_opencv_objdetect=OFF",
        "-D", "BUILD_opencv_photo=OFF",
        "-D", "BUILD_opencv_stitching=OFF",
        "-D", "BUILD_opencv_superres=OFF",
        "-D", "BUILD_opencv_ts=OFF",
        "-D", "BUILD_opencv_video=OFF",
        "-D", "BUILD_videoio_plugins=OFF",
        "-D", "BUILD_opencv_videostab=OFF",
        "-D", "WITH_IPP=ON",
        "-D", "WITH_MKL=ON",        # 启用 Intel MKL
        "-D", "MKL_USE_TBB=ON",
        "-D", "WITH_TBB=ON",
        "-D", "BUILD_TBB=ON",
        "-D", "WITH_TURBOJPEG=ON",
        "-D", "WITH_LAPACK=ON",
        "-D", "WITH_BLAS=ON",
        ".."
    ], check=True)
    subprocess.run(["make", "-j4"], check=True)
    subprocess.run(["make", "install"], check=True)

    # Clean up
    os.chdir("../..")
    shutil.rmtree(OPENCV_DIR)
    os.remove(OPENCV_ZIP)
    
    OPENCV_INCLUDE = os.path.join(install_dir, "include/opencv4")
    OPENCV_LIB = os.path.join(install_dir, "lib")

    print(f"OpenCV installation complete. Please manully set OPENCV_INCLUDE={OPENCV_INCLUDE} and OPENCV_LIB={OPENCV_LIB} in your environment.")

    return OPENCV_INCLUDE, OPENCV_LIB

if __name__ == "__main__":
    # Download and build OpenCV if needed
    # download_and_build_opencv()

    import fire
    fire.Fire(download_and_build_opencv)
