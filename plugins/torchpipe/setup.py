import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Optional
    
import torch
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils import log
from distutils.util import get_platform
from distutils.command.clean import clean as CleanCommand
from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

from setuptools import dist
dist.Distribution().fetch_build_eggs(["omniback"])

def trt_inc_dir():
    incs = ["/usr/include/aarch64-linux-gnu", '/usr/include/x86_64-linux-gnu/',
            "/usr/include", "/usr/local/include"]
    inc = os.getenv("TENSORRT_INCLUDE")
    if inc is not None:
        incs.insert(0, inc)
    for idir in incs:
        if os.path.exists(os.path.join(idir, "NvInfer.h")):
            return idir
    raise RuntimeError(
        "TensorRT include directory not found. Set TENSORRT_INCLUDE environment variable to specify its location.")


def cv_inc_dir():
    incs = ["/usr/include/aarch64-linux-gnu/opencv4",
            "/usr/include/opencv4", "/usr/local/include/opencv4", os.path.expanduser("~/opencv_install/include/opencv4")]
    inc = os.getenv("OPENCV_INCLUDE")
    if inc is not None:
        incs.insert(0, inc)
    for idir in incs:
        if os.path.exists(os.path.join(idir, "opencv2/core.hpp")):
            return idir
    raise RuntimeError(
        "OpenCV include directory not found. Set OPENCV_INCLUDE environment variable to specify its location.")


def trt_lib_dir():
    libs = ["/usr/lib/aarch64-linux-gnu", '/usr/lib/x86_64-linux-gnu/']
    lib = os.getenv("TENSORRT_LIB")
    if lib is not None:
        libs.insert(0, lib)
    else:
        ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        if ld_lib_path is not None:
            # 分割路径并添加到搜索列表的前面
            ld_paths = ld_lib_path.split(':')
            libs = ld_paths + libs
        libs += ["/usr/lib", "/usr/local/lib"]

    for ldir in libs:
        if ldir and os.path.exists(os.path.join(ldir, "libnvinfer.so")):
            return ldir
    raise RuntimeError(
        "TensorRT library directory not found. Set TENSORRT_LIB environment variable to specify its location.")


def cv_lib_dir():
    libs = ["/usr/lib/aarch64-linux-gnu", os.path.expanduser("~/opencv_install/lib")]
    lib = os.getenv("OPENCV_LIB")
    if lib is not None:
        libs.insert(0, lib)
    else:
        ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        if ld_lib_path is not None:
            ld_paths = ld_lib_path.split(':')
            libs = ld_paths + libs
        libs += ["/usr/lib", "/usr/local/lib"]

    for ldir in libs:
        if ldir and os.path.exists(os.path.join(ldir, "libopencv_core.so")):
            return ldir
    raise RuntimeError(
        "OpenCV library directory not found. Set OPENCV_LIB environment variable to specify its location.")



            
class Config:
    def __init__(self):
        import omniback
        
        self.root_dir = Path(__file__).absolute().parent
        self.csrc_dir = self.root_dir / "torchpipe/csrc"

        # Build flags
        self.debug = os.getenv("DEBUG", "0") == "1"

        # Omniback configuration
        self.omniback_includes = omniback.libinfo.include_paths()
        self.omniback_lib_dir = os.path.dirname(omniback.libinfo.find_libomniback()) 
        # self.omniback_c_so = omniback._C.__file__


        # Print config
        log.info("\nTorchpipe Build Configuration:")
        for k, v in vars(self).items():
            if not k.startswith("_"):
                log.info(f"{k.upper():<20}: {v}")


config = Config()


class BuildHelper:
    @staticmethod
    def get_compile_args() -> Tuple[List[Tuple[str, None]], Dict[str, List[str]]]:
        compile_args = {
            "cxx": ["-std=c++17", "-Wno-sign-compare", "-Wno-deprecated-declarations", "-Wno-reorder"]
        }

        if config.debug:
            compile_args["cxx"] += ["-g", "-O0", "-Wall", "-Werror"]
        else:
            compile_args["cxx"] += ["-O2", "-g0"]
        
        key = os.getenv("SECRET_KEY")
        if not key:
            import string
            import secrets
            alphabet = string.ascii_letters + string.digits
            key = ''.join(secrets.choice(alphabet) for _ in range(48))
            print(f'Using random SECRET_KEY. You can reset it through SECRET_KEY environment variable')
        print(
            f'You are using a temporary encryption implementation. Users should replace this with a more secure approach.')

        compile_args["cxx"] += [f"-DSECRET_KEY={key}"]

        compile_args["nvcc"] = []

        return [], compile_args

    @classmethod
    def create_extension(cls, name: str, sources: List[str], include_dirs:List[str] =[], lib_dirs: List[str] =[], ** kwargs):
        if not isinstance(sources, list):
            sources = list(sources)

        base_params = {
            "include_dirs": [
                config.csrc_dir,
                str(Path(CUDA_HOME) / "include/")
            ] + config.omniback_includes + include_dirs,
            "library_dirs": [
                config.omniback_lib_dir,
                str(Path(CUDA_HOME) / "lib64/")
            ]+lib_dirs,
            "libraries": ["omniback"],  # 基类默认库
            "extra_link_args": [
            ] + [
                 *kwargs.get("extra_link_args", []),
                 '-Wl,--no-as-needed',
                 "-Wl,-rpath,$ORIGIN",
                 ]
        }

        list_params = ["include_dirs", "library_dirs",
                       "libraries", "extra_link_args"]

        merged_params = {}
        for key in base_params:
            if key in list_params:
                base_list = base_params[key]
                user_list = kwargs.get(key, [])
                merged_params[key] = list(
                    dict.fromkeys([*base_list, *user_list]))
            else:
                merged_params[key] = kwargs.get(key, base_params[key])

        for key in kwargs:
            if key not in merged_params:
                merged_params[key] = kwargs[key]

        define_macros, extra_compile_args = BuildHelper.get_compile_args()
        if 'define_macros' not in merged_params:
            merged_params['define_macros'] = define_macros
        else:
            merged_params['define_macros'] += define_macros

        merged_params['extra_compile_args'] = extra_compile_args

        sources = [str(x) for x in sources]
        ext = CUDAExtension(
            name=f"torchpipe.{name}",
            sources=sources,
            **merged_params
        )

        return ext


def build_core_extension():
    sources = [
        *config.csrc_dir.glob("ffi/*.cpp"),
        *config.csrc_dir.glob("torchplugins/*.cpp"),
        *config.csrc_dir.glob("helper/*.cpp"),
        # *config.csrc_dir.glob("pybind/*.cpp"),
    ]
    sources = [str(x) for x in sources]
    sources += config.csrc_dir.glob("cuda/*.cu")

    return BuildHelper.create_extension(
        name="native",
        sources=sources,
    )


def build_nvjpeg_extension():
    sources = config.csrc_dir.glob("nvjpeg_torch/*.cpp")
    sources = [str(x) for x in sources]
    return BuildHelper.create_extension(
        name="image",
        sources=sources,
        libraries=["nvjpeg"],
        define_macros=[("NVJPEG_FOUND", 1)],
    )


def build_opencv_extension():
    sources = config.csrc_dir.glob("mat_torch/*.cpp")

    opencv_libs = ["opencv_core", "opencv_imgproc", "opencv_imgcodecs"]

    for lib in opencv_libs:
        assert any(Path(cv_lib_dir()).glob(f"lib{lib}.so*")), \
            f"OpenCV library {lib} not found in {cv_lib_dir()}"

    sources = [str(x) for x in sources]
    return BuildHelper.create_extension(
        name="mat",
        sources=sources,
        libraries=opencv_libs,
        include_dirs=[cv_inc_dir()],
        lib_dirs=[cv_lib_dir()],
        define_macros=[("BUILD_CV2", 1)],
    )


def build_trt_extension():
    sources = config.csrc_dir.glob("tensorrt_torch/*.cpp")

    return BuildHelper.create_extension(
        name="trt",
        sources=sources,
        include_dirs=[trt_inc_dir()],
        lib_dirs=[trt_lib_dir()],
        libraries=["nvinfer", "nvinfer_plugin", "nvonnxparser"],
        define_macros=[("BUILD_TRT", 1)],
    )

if __name__ == "__main__":
    
    
    extensions = [
        build_core_extension(),
    ]
    
    if not '--no-trt' in sys.argv:
        extensions.append(build_trt_extension())
    else:
        sys.argv.remove('--no-trt')
    if not '--no-nvjpeg' in sys.argv:
        extensions.append(build_nvjpeg_extension())
    else:
        sys.argv.remove('--no-nvjpeg')
        
    if '--cv2' in sys.argv:
        extensions.append(build_opencv_extension())
        sys.argv.remove('--cv2')
        
    setup(
        name="torchpipe",
        version="0.9.0",
        author="torchpipe Team",
        setup_requires=['omniback', 'torch'],
        install_requires=['omniback'],
        description="High-performance inference pipeline for PyTorch",
        packages=find_packages(exclude=("test",)),
        package_data={
            "torchpipe": ["*.so*", "*.pyi", "py.typed"],
        },
        ext_modules=extensions,
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        },
        python_requires=">=3.8",
    )
