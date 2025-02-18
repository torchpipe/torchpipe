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

import torch
from pkg_resources import DistributionNotFound, get_distribution, parse_version
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME, CUDAExtension, ROCM_HOME

import hami

FORCE_CUDA = os.getenv("FORCE_CUDA", "1") == "1"
FORCE_MPS = os.getenv("FORCE_MPS", "0") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"
USE_PNG = os.getenv("TORCHPIPE_USE_PNG", "0") == "1"
USE_JPEG = os.getenv("TORCHPIPE_USE_JPEG", "0") == "1"
USE_WEBP = os.getenv("TORCHPIPE_USE_WEBP", "0") == "1"
USE_NVJPEG = os.getenv("TORCHPIPE_USE_NVJPEG", "1") == "1"


USE_OPENCV = os.getenv("TORCHPIPE_USE_OPENCV", "1") == "1"

OPENCV_INCLUDE = os.getenv("OPENCV_INCLUDE", "/usr/local/include/opencv4/")
OPENCV_LIB = os.getenv("OPENCV_LIB", "/usr/local/lib/")

USE_TENSORRT = os.getenv("TORCHPIPE_USE_TENSORRT", "1") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", None)
# Note: the GPU video decoding stuff used to be called "video codec", which
# isn't an accurate or descriptive name considering there are at least 2 other
# video deocding backends in torchpipe. I'm renaming this to "gpu video
# decoder" where possible, keeping user facing names (like the env var below) to
# the old scheme for BC.
USE_GPU_VIDEO_DECODER = os.getenv("TORCHPIPE_USE_VIDEO_CODEC", "0") == "1"
# Same here: "use ffmpeg" was used to denote "use cpu video decoder".
USE_CPU_VIDEO_DECODER = os.getenv("TORCHPIPE_USE_FFMPEG", "0") == "1"

TORCHPIPE_INCLUDE = os.environ.get("TORCHPIPE_INCLUDE", "")
TORCHPIPE_LIBRARY = os.environ.get("TORCHPIPE_LIBRARY", "")
TORCHPIPE_INCLUDE = TORCHPIPE_INCLUDE.split(os.pathsep) if TORCHPIPE_INCLUDE else []
TORCHPIPE_LIBRARY = TORCHPIPE_LIBRARY.split(os.pathsep) if TORCHPIPE_LIBRARY else []

ROOT_DIR = Path(__file__).absolute().parent
CSRS_DIR = ROOT_DIR / "torchpipe/csrc"

BUILD_CUDA_SOURCES = (torch.cuda.is_available() and ((CUDA_HOME is not None) )) or FORCE_CUDA

package_name = os.getenv("TORCHPIPE_PACKAGE_NAME", "torchpipe")
HAMI_INCLUDES = hami.get_includes()
HAMI_library_dirs = [hami.get_library_dir()]
print("Torchpipe build configuration:")
print(f"{FORCE_CUDA = }")
print(f"{DEBUG = }")
print(f"{USE_PNG = }")
print(f"{USE_JPEG = }")
print(f"{USE_WEBP = }")
print(f"{USE_NVJPEG = }")
print(f"{NVCC_FLAGS = }")
print(f"{TORCHPIPE_INCLUDE = }")
print(f"{TORCHPIPE_LIBRARY = }")
print(f"{BUILD_CUDA_SOURCES = }")
print(f"{HAMI_INCLUDES = }")
print(f"{HAMI_library_dirs = }")


def get_version():
    with open(ROOT_DIR / "version.txt") as f:
        version = f.readline().strip()
    sha = "Unknown"

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        pass

    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    return version, sha


def write_version_file(version, sha):
    # Exists for BC, probably completely useless.
    with open(ROOT_DIR / "torchpipe/version.py", "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write("from torchpipe.extension import _check_cuda_version\n")
        f.write("if _check_cuda_version() > 0:\n")
        f.write("    cuda = _check_cuda_version()\n")


def get_requirements():
    def get_dist(pkgname):
        try:
            return get_distribution(pkgname)
        except DistributionNotFound:
            return None

    pytorch_dep = os.getenv("TORCH_PACKAGE_NAME", "torch")
    if os.getenv("PYTORCH_VERSION"):
        pytorch_dep += "==" + os.getenv("PYTORCH_VERSION")

    requirements = [
        "numpy",
        pytorch_dep,
    ]

    return requirements


def get_macros_and_flags():
    define_macros = []
    extra_compile_args = {"cxx": []}
    if BUILD_CUDA_SOURCES:
        define_macros += [("WITH_CUDA", None)]
        if NVCC_FLAGS is None:
            nvcc_flags = []
        else:
            nvcc_flags = NVCC_FLAGS.split(" ")
        extra_compile_args["nvcc"] = nvcc_flags

    if DEBUG:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-O0")
        if "nvcc" in extra_compile_args:
            # we have to remove "-OX" and "-g" flag if exists and append
            nvcc_flags = extra_compile_args["nvcc"]
            extra_compile_args["nvcc"] = [f for f in nvcc_flags if not ("-O" in f or "-g" in f)]
            extra_compile_args["nvcc"].append("-O0")
            extra_compile_args["nvcc"].append("-g")
    else:
        extra_compile_args["cxx"].append("-g0")
    extra_compile_args['cxx']+=["-std=c++17"]
    return define_macros, extra_compile_args


def make_C_extension():
    print("Building _C extension")

    sources = (
        list(CSRS_DIR.glob("*.cpp"))
        + list(CSRS_DIR.glob("torchplugins/*.cpp"))
        + list(CSRS_DIR.glob("helper/*.cpp"))
        + list(CSRS_DIR.glob("pybind/*.cpp"))
    )
    
    cuda_sources = list(CSRS_DIR.glob("cuda/*.cu"))
    

    if BUILD_CUDA_SOURCES:
        Extension = CUDAExtension
        sources += cuda_sources
    else:
        Extension = CppExtension
        
    define_macros, extra_compile_args = get_macros_and_flags()
    
    hami_C_so = hami.get_C_path()
    # hami_C_dir = os.path.dirname(hami_C_so)

    assert os.path.exists(hami_C_so)
    
    import glob
    # hami_lib = glob.glob(os.path.join(hami.get_library_dir(), "libhami.*"))[0]
    
    return Extension(
        name="torchpipe.native",
        sources=sorted(str(s) for s in sources),
        include_dirs=[CSRS_DIR] + HAMI_INCLUDES,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        library_dirs=HAMI_library_dirs,
        libraries = ['hami'],
        extra_link_args=[
            f'-Wl,-rpath,$ORIGIN/../../hami',
            f'-Wl,-rpath,{os.path.dirname(hami_C_so)}',
            '-Wl,--no-as-needed',
            hami_C_so,  # 链接 _C.*.so
        ],
    )



def find_library(header):
    # returns (found, include dir, library dir)
    # if include dir or library dir is None, it means that the library is in
    # standard paths and don't need to be added to compiler / linker search
    # paths

    searching_for = f"Searching for {header}"

    for folder in TORCHPIPE_INCLUDE:
        if (Path(folder) / header).exists():
            print(f"{searching_for} in {Path(folder) / header}. Found in TORCHPIPE_INCLUDE.")
            return True, None, None
    print(f"{searching_for}. Didn't find in TORCHPIPE_INCLUDE.")

    # Try conda-related prefixes. If BUILD_PREFIX is set it means conda-build is
    # being run. If CONDA_PREFIX is set then we're in a conda environment.
    for prefix_env_var in ("BUILD_PREFIX", "CONDA_PREFIX"):
        if (prefix := os.environ.get(prefix_env_var)) is not None:
            prefix = Path(prefix)
            if sys.platform == "win32":
                prefix = prefix / "Library"
            include_dir = prefix / "include"
            library_dir = prefix / "lib"
            if (include_dir / header).exists():
                print(f"{searching_for}. Found in {prefix_env_var}.")
                return True, str(include_dir), str(library_dir)
        print(f"{searching_for}. Didn't find in {prefix_env_var}.")

    if sys.platform == "linux":
        for prefix in (Path("/usr/include"), Path("/usr/local/include")):
            if (prefix / header).exists():
                print(f"{searching_for}. Found in {prefix}.")
                return True, None, None
            print(f"{searching_for}. Didn't find in {prefix}")

    return False, None, None


def make_image_extension():
    print("Building image extension")

    include_dirs = TORCHPIPE_INCLUDE.copy()
    library_dirs = TORCHPIPE_LIBRARY.copy()
    
    include_dirs += HAMI_INCLUDES

    libraries = []
    define_macros, extra_compile_args = get_macros_and_flags()

    sources = (
        list(CSRS_DIR.glob("nvjpeg_torch/*.cpp"))
    )
    

    Extension = CppExtension
    
    assert torch.cuda.is_available()
    if USE_NVJPEG:
        nvjpeg_found = CUDA_HOME is not None and (Path(CUDA_HOME) / "include/nvjpeg.h").exists()

        if nvjpeg_found:
            print("Building torchpipe with NVJPEG image support")
            libraries.append("nvjpeg")
            define_macros += [("NVJPEG_FOUND", 1)]
            Extension = CUDAExtension
        else:
            # warnings.warn("Building torchpipe without NVJPEG support")
            raise RuntimeError("NVJPEG not found. You may need to set CUDA_HOME environment variable.")
    else:
        warnings.warn("Building torchpipe without NVJPEG support")

    return Extension(
        name="torchpipe.image",
        sources=sorted(str(s) for s in sources),
        include_dirs=include_dirs + [CSRS_DIR],
        library_dirs=HAMI_library_dirs,
        libraries = ['hami']+libraries,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )



class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)



def make_mat_extension():

    print("Building image extension")

    include_dirs = TORCHPIPE_INCLUDE.copy()
    library_dirs = TORCHPIPE_LIBRARY.copy()
    
    include_dirs += HAMI_INCLUDES

    libraries = []
    define_macros, extra_compile_args = get_macros_and_flags()

    sources = (
        list(CSRS_DIR.glob("mat_torch/*.cpp"))
    )

    Extension = CppExtension
    
    if USE_OPENCV:
        global OPENCV_INCLUDE, OPENCV_LIB
        opencv_found =  (Path(OPENCV_INCLUDE) / "opencv2/core.hpp").exists() and ( Path(OPENCV_LIB) / "libopencv_core.so").exists()
        
        if not opencv_found:
            warnings.warn("not found opencv. please set OPENCV_INCLUDE and  OPENCV_LIB.")
            warnings.warn("Auto download and build Opencv now.")
            from download_and_build_opencv import download_and_build_opencv

            OPENCV_INCLUDE, OPENCV_LIB = download_and_build_opencv()
            os.environ["OPENCV_INCLUDE"] = OPENCV_INCLUDE
            os.environ["OPENCV_LIB"] = OPENCV_LIB
            print("new OPENCV_INCLUDE={OPENCV_INCLUDE} OPENCV_LIB={OPENCV_LIB}")
        print("Building torchpipe with opencv backends support")
        libraries +=["opencv_core", "opencv_imgproc", "opencv_imgcodecs"]
        define_macros += [("OPENCV_FOUND", 1)]
        include_dirs += [OPENCV_INCLUDE]
        library_dirs  += [OPENCV_LIB]
            # Extension = CUDAExtension
        
    else:
        warnings.warn("Building torchpipe without Opencv support")

    return Extension(
        name="torchpipe.mat",
        sources=sorted(str(s) for s in sources),
        include_dirs=include_dirs + [CSRS_DIR],
        library_dirs=HAMI_library_dirs + library_dirs,
        libraries = ['hami']+libraries,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    version, sha = get_version()
    write_version_file(version, sha)

    print(f"Building wheel {package_name}-{version}")

    with open("README.md") as f:
        readme = f.read()

    extensions = [
        make_C_extension(),
        make_image_extension(),
        make_mat_extension(),
        # *make_video_decoders_extensions(),
    ]

    setup(
        name=package_name,
        version=version,
        author="Hami/torchpipe Team",
        author_email="",
        url="https://github.com/torchpipe/torchpipe",
        description="image and video datasets and models for torch deep learning",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="Apache License, Version 2.0",
        packages=find_packages(exclude=("test",)),
        package_data={package_name: ["*.dll", "*.dylib", "*.so", "prototype/datasets/_builtin/*.categories"]},
        zip_safe=False,
        install_requires=get_requirements(),
        extras_require={
            "torch": ["torch>=1.10.0"],
            "hami":"hami>=0.0.1"
        },
        ext_modules=extensions,
        python_requires=">=3.7",
        cmdclass={
            "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            "clean": clean,
        },
    )