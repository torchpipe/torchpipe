import hami
import os
import sys
import glob
import shutil
import warnings
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from distutils import log
from distutils.util import get_platform
from distutils.command.clean import clean as CleanCommand
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

# Environment Configuration
class Config:
    def __init__(self):
        self.root_dir = Path(__file__).absolute().parent
        self.csrc_dir = self.root_dir / "torchpipe/csrc"
        
        # Build flags
        self.force_cuda = os.getenv("FORCE_CUDA", "1") == "1"
        self.debug = os.getenv("DEBUG", "0") == "1"
        self.use_nvjpeg = os.getenv("TORCHPIPE_USE_NVJPEG", "1") == "1"
        
        # Dependency paths
        self.opencv_include = os.getenv("OPENCV_INCLUDE", "/usr/local/include/opencv4/")
        self.opencv_lib = os.getenv("OPENCV_LIB", "/usr/local/lib/")
        self.tensorrt_include = os.getenv("TENSORRT_INCLUDE", "/usr/local/include/")
        self.tensorrt_lib = os.getenv("TENSORRT_LIB", "/usr/local/lib/")
        
        # Hami configuration
        self.hami_includes = hami.get_includes()
        self.hami_lib_dir = hami.get_library_dir()
        self.hami_c_so = hami.get_C_path()
        
        # System checks
        self.cuda_available = torch.cuda.is_available() and CUDA_HOME is not None
        self.build_cuda = self.cuda_available or self.force_cuda

        # Print config
        log.info("\nTorchpipe Build Configuration:")
        for k, v in vars(self).items():
            if not k.startswith("_"):
                log.info(f"{k.upper():<20}: {v}")

config = Config()

# Version Handling
def get_version() -> Tuple[str, str]:
    version = (config.root_dir / "version.txt").read_text().strip()
    sha = "Unknown"
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], 
                                    cwd=str(config.root_dir)).decode().strip()
    except Exception:
        pass
    
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += f"+{sha[:7]}"
    
    return version, sha

def write_version_file(version: str, sha: str):
    content = f"""__version__ = '{version}'
git_version = {repr(sha)}
from torchpipe.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
"""
    (config.root_dir / "torchpipe/version.py").write_text(content)

# Dependency Management
class DependencyManager:
    @staticmethod
    def validate_library(header: str, include_dir: Path, lib_dir: Path) -> bool:
        return (include_dir / header).exists() and any(lib_dir.glob(f"*{header.split('.')[0]}*"))

    @classmethod
    def handle_opencv(cls):
        if not cls.validate_library("opencv2/core.hpp", 
                                  Path(config.opencv_include), 
                                  Path(config.opencv_lib)):
            log.warn("OpenCV not found. Attempting to install...")
            from download_and_build_opencv import download_and_build_opencv
            new_include, new_lib = download_and_build_opencv()
            config.opencv_include = new_include
            config.opencv_lib = new_lib
            # raise RuntimeError(new_include, new_lib, new_lib)
            # import pdb; pdb.set_trace()

    @classmethod
    def handle_tensorrt(cls):
        if not cls.validate_library("NvInfer.h",
                                  Path(config.tensorrt_include),
                                  Path(config.tensorrt_lib)):
            log.warn("TensorRT not found. Attempting to install...")
            from download_and_install_tensorrt import download_and_install_trt
            new_include, new_lib = download_and_install_trt()
            config.tensorrt_include = new_include
            config.tensorrt_lib = new_lib

# Build Helpers
class BuildHelper:
    @staticmethod
    def get_compile_args() -> Tuple[List[Tuple[str, None]], Dict[str, List[str]]]:
        macros = [("WITH_CUDA", None)] if config.build_cuda else []
        
        compile_args = {
            "cxx": ["-std=c++17", "-Wno-sign-compare", "-Wno-deprecated-declarations"]
        }
        
        if config.debug:
            compile_args["cxx"] += ["-g", "-O0", "-Wall", "-Werror"]
        else:
            compile_args["cxx"] += ["-O2", "-g0"]
            
        if config.build_cuda:
            nvcc_flags = os.getenv("NVCC_FLAGS", "").split()
            if config.debug:
                nvcc_flags = [f for f in nvcc_flags if "-O" not in f and "-g" not in f]
                nvcc_flags += ["-O0", "-g"]
            compile_args["nvcc"] = nvcc_flags
            
        return macros, compile_args

    @classmethod
    def create_extension(cls, name: str, sources: List[str], **kwargs):
        if not isinstance(sources, list):
            sources = list(sources)
        extra_path = []
        if name != "native":
            native_so_path = os.path.join(os.path.dirname(__file__), "torchpipe/native.so")
            if not os.path.exists(native_so_path):
                build_lib_dir = Path(os.getcwd()) / "build" / f"lib.{get_platform()}-cpython-{sys.version_info.major}{sys.version_info.minor}"
                native_so_rel_path = Path("torchpipe") / "native.so"
                native_so_path = build_lib_dir / native_so_rel_path

            extra_path = [native_so_path]
        base_params = {
            "include_dirs": [
                config.csrc_dir, 
                config.opencv_include,
                config.tensorrt_include, 
                str(Path(CUDA_HOME) / "include/")
            ] + config.hami_includes,
            "library_dirs": [
                config.hami_lib_dir, 
                config.opencv_lib,
                config.tensorrt_lib,
                str(Path(CUDA_HOME) / "lib64/")
            ],
            "libraries": ["hami"],  # 基类默认库
            "extra_link_args": [
                os.path.join(hami.get_library_dir(), "libhami.so"),
                config.hami_c_so]
                + extra_path
                + [ f'-Wl,-rpath,{config.opencv_lib}', 
                f'-Wl,-rpath,{config.tensorrt_lib}', 
                f'-Wl,-rpath,$ORIGIN',
                *kwargs.get("extra_link_args", []),
                '-Wl,--no-as-needed'
            ]
        }

        list_params = ["include_dirs", "library_dirs", "libraries", "extra_link_args"]

        merged_params = {}
        for key in base_params:
            if key in list_params:
                base_list = base_params[key]
                user_list = kwargs.get(key, [])
                merged_params[key] = list(dict.fromkeys([*base_list, *user_list]))
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
        # CppExtension
        return CUDAExtension(
            name=f"torchpipe.{name}",
            sources=sources,
            **merged_params
        )

# Extension Builders
def build_core_extension():
    sources = [
        *config.csrc_dir.glob("*.cpp"),
        *config.csrc_dir.glob("torchplugins/*.cpp"),
        *config.csrc_dir.glob("helper/*.cpp"),
        *config.csrc_dir.glob("pybind/*.cpp"),
    ]
    if config.build_cuda:
        sources += config.csrc_dir.glob("cuda/*.cu")
    
    return BuildHelper.create_extension(
        name="native",
        sources=sources,
    )
    

def build_nvjpeg_extension():
    DependencyManager.handle_opencv()
    
    sources = config.csrc_dir.glob("nvjpeg_torch/*.cpp")
    
            
    return BuildHelper.create_extension(
        name="image",
        sources=sources,
        libraries=["nvjpeg"],
        define_macros=[("NVJPEG_FOUND", 1)],
    )

def build_opencv_extension():
    DependencyManager.handle_opencv()
    
    sources = config.csrc_dir.glob("mat_torch/*.cpp")
    
    opencv_libs = ["opencv_core", "opencv_imgproc", "opencv_imgcodecs"]
    for lib in opencv_libs:
        assert any(Path(config.opencv_lib).glob(f"lib{lib}.so*")), \
            f"OpenCV library {lib} not found in {config.opencv_lib}"
            
    return BuildHelper.create_extension(
        name="mat",
        sources=sources,
        libraries=opencv_libs,
        define_macros=[("OPENCV_FOUND", 1)],
    )
    
def build_trt_extension():
    DependencyManager.handle_tensorrt()
    sources = config.csrc_dir.glob("tensorrt_torch/*.cpp")
    
    return BuildHelper.create_extension(
        name="trt",
        sources=sources,
        libraries=["nvinfer", "nvinfer_plugin", "nvonnxparser" ],
        define_macros=[("TENSORRT_FOUND", 1)],
    )

# Setup Main
if __name__ == "__main__":
    # Validate environment
    assert os.path.exists(config.hami_c_so), "Hami C extension not found!"
    
    # Prepare version
    version, sha = get_version()
    write_version_file(version, sha)
    
    # Build extensions
    extensions = [
        build_core_extension(),
        build_opencv_extension(),
        build_nvjpeg_extension(),
        build_trt_extension(),
    ]

    # Setup configuration
    setup(
        name=os.getenv("TORCHPIPE_PACKAGE_NAME", "torchpipe"),
        version=version,
        author="Hami/torchpipe Team",
        description="High-performance inference pipeline for PyTorch",
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/torchpipe/torchpipe",
        packages=find_packages(exclude=("test",)),
        ext_modules=extensions,
        install_requires=[
            "numpy",
            "hami-core",
        ],
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": CleanCommand,
        },
        zip_safe=False,
        python_requires=">=3.8",
    )