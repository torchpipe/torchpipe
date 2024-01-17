# Copyright 2021-2023 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils import cpp_extension
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension
from pkg_resources import parse_version, get_distribution, DistributionNotFound
import distutils.command.clean
import distutils.spawn
import glob
import os
import shutil
import subprocess
import sys

from distutils.version import StrictVersion

# todo modified acc. to https://github.com/open-mmlab/mmdeploy/blob/master/setup.py

import torch

if StrictVersion(torch.__version__.split("+")[0]) < "1.10.0a0":
    raise RuntimeError(
        f"torch's version should >= 1.10.0a0, but get {torch.__version__}")

if sys.version_info < (3, 7, 0):
    raise RuntimeError("TorchPipe requires Python 3.7.0 or later.")

torch_vision = torch.__version__.split("+")[0]


__cuda_version__ = torch.version.cuda

# https://stackoverflow.com/questions/8106258/cc1plus-warning-command-line-option-wstrict-prototypes-is-valid-for-ada-c-o
if True:
    # remove -g setting by /usr/lib/python3.8/config-3.8-x86_64-linux-gnu/Makefile
    import os
    from distutils.sysconfig import get_config_vars

    target = ["-g ", "-Wstrict-prototypes", "-Wdeprecated-declarations"]
    (opt,) = get_config_vars("OPT")
    os.environ["OPT"] = " ".join(
        flag for flag in opt.split() if flag not in target)
    cfg_vars = distutils.sysconfig.get_config_vars()

    for key, value in cfg_vars.items():
        if type(value) == str:
            for i in target:
                value = value.replace(i, " ")
            cfg_vars[key] = value


def read(*names, **kwargs):
    with open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None


cwd = os.path.dirname(os.path.abspath(__file__))

version_txt = os.path.join(cwd, "version.txt")
with open(version_txt) as f:
    version = f.readline().strip()
sha = "Unknown"
package_name = "torchpipe"

try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    pass


if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]


WITH_OPENCV = os.getenv("WITH_OPENCV", "1") == "1"
WITH_TORCH = os.getenv("WITH_TORCH", "1") == "1"
WITH_TENSORRT = os.getenv("WITH_TENSORRT", "1") == "1"

BUILD_PPLCV = os.getenv("BUILD_PPLCV", "0") == "1"
PPLCV_INSTALL = os.getenv("PPLCV_INSTALL", "0") == "1"

WITH_CVCUDA = os.getenv("WITH_CVCUDA", "-1") == "1"
FORCE_CVCUDA_DIR = os.getenv("FORCE_CVCUDA_DIR", None)

CVCUDA_INSTALL=None

if not WITH_CVCUDA and not FORCE_CVCUDA_DIR :
    CVCUDA_INSTALL = None
else:
    WITH_CVCUDA = True

    if FORCE_CVCUDA_DIR:
        CVCUDA_INSTALL = FORCE_CVCUDA_DIR
    else:
        target_dir = os.path.join(os.path.expanduser("~"), ".cache/nvcv/")
        CVCUDA_INSTALL = os.path.join(target_dir, "opt/nvidia/cvcuda0/")
        properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if properties.major < 6:
            raise RuntimeError("CVCUDA only support cuda >= 6.1")
        elif properties.major == 6:
            print("Warning: You are using CV-CUDA with architecture sm6.1, which is not officially supported. We will utilize a self-compiled CVCUDA library.")
            subprocess.check_output(["python", "torchpipe/tool/get_cvcuda.py", "--sm61"])
        else:
            subprocess.check_output(["python", "torchpipe/tool/get_cvcuda.py"])
        if not os.path.exists(CVCUDA_INSTALL):
            raise RuntimeError(f"WITH_CVCUDA: {CVCUDA_INSTALL} not found.")
# if not PPLCV_INSTALL and not BUILD_PPLCV:
#     BUILD_PPLCV = True

if BUILD_PPLCV:
    if not PPLCV_INSTALL:
        PPLCV_INSTALL = os.path.abspath(os.path.join("build", "install"))

IPIPE_KEY = os.getenv("IPIPE_KEY", None)
if IPIPE_KEY is None:
    import random
    import time
    IPIPE_KEY = str(random.randint(0, 1000000000) + time.time())
    print("WARN: random IPIPE_KEY used.")
else:
    print("IPIPE_KEY setted.")

if cpp_extension.CUDA_HOME is None:
    raise RuntimeError("`CUDA_HOME` is undefined. Please check if you got CUDA installed or CUDA_HOME exported.")

raw_output = subprocess.check_output(
    [cpp_extension.CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
)
output = raw_output.split()
build_cuda_version = output[output.index("release") + 1].strip(",").split(".")

raw_output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)

output = raw_output.split()
build_driver_version = output[output.index("Driver") + 2].strip().split(".")

# https://www.cnblogs.com/phillee/p/12049208.html
if os.getenv("TORCH_CUDA_ARCH_LIST") is None:

    avaliable = {}
    avaliable["8.9"] = False
    avaliable["8.6"] = False
    avaliable["8.0"] = False
    # https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html
    # https://en.m.wikipedia.org/wiki/CUDA#GPUs_supported
    if int(build_cuda_version[0]) == 12:
        avaliable["8.9"] = True
        avaliable["8.6"] = True
        avaliable["8.0"] = True
    elif int(build_cuda_version[0]) == 11:
        avaliable["8.9"] = avaliable["8.9"] or int(__cuda_version__.split(".")[1])>=8
        avaliable["8.6"] = avaliable["8.6"] or int(__cuda_version__.split(".")[1])>=1
        avaliable["8.0"] = avaliable["8.0"] or int(__cuda_version__.split(".")[1])>=0

    if avaliable["8.9"]:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
    elif avaliable["8.6"]:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    elif avaliable["8.0"]:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0+PTX"
    else:
        assert int(build_cuda_version[0]) == 10
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5"


class CMakeExtension(Extension):
    def __init__(self, name, cmakelist_dir):
        Extension.__init__(self, name, sources=[])
        self.cmakelist_dir = os.path.abspath(cmakelist_dir)


class CMakeBuild(BuildExtension.with_options(no_python_abi_suffix=True)):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def run(self):
        cmake_extensions = [
            e for e in self.extensions if isinstance(e, CMakeExtension)]
        self.extensions = [
            e for e in self.extensions if not isinstance(e, CMakeExtension)]

        for ext in cmake_extensions:
            self.build_cmake_extension(ext)
        super().run()

    def build_cmake_extension(self, ext: CMakeExtension) -> None:
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        build_directory = os.path.abspath(self.build_temp)
        if not os.path.exists(build_directory):
            os.makedirs(build_directory)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_directory,
            # '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        install_dir = PPLCV_INSTALL

        cmake_args += ['-DCMAKE_BUILD_TYPE=Release', "-DPPLCV_USE_CUDA=ON",
                       f"-DCMAKE_INSTALL_PREFIX={install_dir}",]

        build_args = ['-j8', "--config", "Release"]

        self.build_args = build_args

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())

        unsupport_arch = [35, 37, 50, 53]
        cuda_cmake = os.path.join(ext.cmakelist_dir, "cmake/cuda.cmake")
        if os.path.exists(cuda_cmake):
            for arch in unsupport_arch:
                subprocess.check_call(
                    [f"sed -i '/arch=compute_{arch},code=sm_{arch}/d' {cuda_cmake}"], shell=True)

        subprocess.check_call(['cmake', ext.cmakelist_dir] + cmake_args,
                              cwd=self.build_temp, env=env)

        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        cmake_cmd = ['cmake',  '--build', '.',
                     "--target", "install"] + self.build_args

        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

    #  cmake -DCMAKE_BUILD_TYPE=Release  -DPPLCV_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=/tmp/ppl.cv/cuda-build/install .. && cmake --build . -j 8 --config Release && cmake --build . --target install -j 8 --config Release

        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
        pass


def check_sub_modules():
    if not os.path.exists("thirdparty/spdlog"):
        raise RuntimeError(
            "submodule not found, you may forget to run 'git submodule update --init --recursive'"
        )


def write_version_file():
    version_path = os.path.join(cwd, "torchpipe", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write('__cuda_version__ = "' + __cuda_version__ + '"\n')
        f.write('__torch_version__ = "' + torch.__version__ + '"\n')
        # f.write('__cudnn_version__ = "' + __cudnn_version__ + '"\n')
        # f.write('__tensorrt_version__ = "' + __tensorrt_version__ + '"\n')


with open("./requirements.txt", "r") as f:
    requirements = f.read().splitlines()


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torchpipe", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "..", "*.cpp")) + glob.glob(
        os.path.join(extensions_dir, "..", "tool", "*.cpp")
    )
    source_cpu = (
        glob.glob(os.path.join(extensions_dir, "core", "src", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "schedule", "src", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "backend", "src", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "pipeline", "src", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "python", "src", "*.cpp"))
        + glob.glob(os.path.join(extensions_dir, "*.cpp"))
    )

    if WITH_OPENCV:
        source_cpu += glob.glob(os.path.join(extensions_dir,
                                "opencv", "src", "*.cpp"))
    if PPLCV_INSTALL:
        source_cpu += glob.glob(os.path.join(extensions_dir,
                                "ppl.cv", "src", "*.cpp"))

    third_includes = [os.path.join(extensions_dir, "../..", "thirdparty/")]

    thirdpart_lib_dirs = []
    thirdpart_libs = []
    extra_link_args=[]

    if WITH_CVCUDA:
        source_cpu += glob.glob(os.path.join(extensions_dir,
                                "cvcuda", "src", "*.cpp"))
        assert(os.path.exists(CVCUDA_INSTALL))
        third_includes += [os.path.join(CVCUDA_INSTALL, "include")]

        cvcuda_libdir = os.path.join(CVCUDA_INSTALL, "lib/x86_64-linux-gnu/")
        thirdpart_lib_dirs += [cvcuda_libdir]
        thirdpart_libs += ["cvcuda", 'nvcv_types']
        # 添加/opt/nvidia/cvcuda0/lib/x86_64-linux-gnu/ 作为动态库搜索路径
        extra_link_args += [f'-Wl,-rpath={cvcuda_libdir}']

        real_path_libcvcuda = os.path.realpath(os.path.join(cvcuda_libdir, "libcvcuda.so"))
        real_path_libnvcv_types = os.path.realpath(os.path.join(cvcuda_libdir, "libnvcv_types.so"))
        assert(os.path.exists(real_path_libcvcuda))
        install_files.append(real_path_libcvcuda)
        install_files.append(real_path_libnvcv_types)

        

    source_cpu += glob.glob(
        os.path.join(extensions_dir, "thirdpart", "pillow-resize", "*.cpp")
    )

    source_cuda = glob.glob(os.path.join(
        extensions_dir, "backend", "src_cuda", "*.cu"))

    source_cuda += glob.glob(
        os.path.join(extensions_dir, "backend", "src_cuda", "*.cpp")
    )
    source_cuda += glob.glob(
        os.path.join(extensions_dir, "thirdpart", "ppl/cv/cuda", "*.cu")
    )
    source_cuda += glob.glob(
        os.path.join(extensions_dir, "thirdpart", "cvcuda", "*.cu")
    )

    sources = main_file + source_cpu
    extension = CppExtension

    compile_cpp_tests = os.getenv("WITH_CPP_MODELS_TEST", "0") == "1"
    if compile_cpp_tests:
        test_dir = os.path.join(this_dir, "test")
        models_dir = os.path.join(this_dir, "torchpipe", "csrc", "models")
        test_file = glob.glob(os.path.join(test_dir, "*.cpp"))
        source_models = glob.glob(os.path.join(models_dir, "*.cpp"))

        test_file = [os.path.join(test_dir, s) for s in test_file]
        source_models = [os.path.join(models_dir, s) for s in source_models]
        tests = test_file + source_models
        tests_include_dirs = [test_dir, models_dir]

    define_macros = [("PYBIND", None)]

    if int(build_cuda_version[0]) >= 11:
        extra_compile_args = {
            "cxx": ["-Wno-unused-parameter", "-std=c++17"]
        }  # {"cxx": ["-std=c++14"]}
    else:
        extra_compile_args = {"cxx": ["-Wno-unused-parameter", "-std=c++14"]}

    assert torch.cuda.is_available()
    if (torch.cuda.is_available()) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")

        extra_compile_args["nvcc"] = nvcc_flags

    if int(build_cuda_version[0]) >= 11:
        extra_compile_args["nvcc"] += ["-std=c++17"]

    if sys.platform == "win32":
        define_macros += [("TORCHPIPE_EXPORTS", None)]
        define_macros += [("USE_PYTHON", None)]
        extra_compile_args["cxx"].append("/MP")

    debug_mode = os.getenv("DEBUG", "0") == "1"

    use_encrypt = os.getenv("USE_DECRYPT", "0") == "1"
    if use_encrypt:
        extra_compile_args["cxx"] += [("-DUSE_DECRYPT")]
    if WITH_TORCH:
        extra_compile_args["cxx"] += [("-DWITH_TORCH")]
    if WITH_OPENCV:
        extra_compile_args["cxx"] += [("-DWITH_OPENCV")]
    if WITH_TENSORRT:
        extra_compile_args["cxx"] += [("-DWITH_TENSORRT")]
    if IPIPE_KEY:
        extra_compile_args["cxx"] += [(f"-DIPIPE_KEY={IPIPE_KEY}")]
    if PPLCV_INSTALL:
        extra_compile_args["cxx"] += [("-DWITH_PPLCV")]
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", "0") == "1":
        extra_compile_args["cxx"] += [("-DPYTORCH_NO_CUDA_MEMORY_CACHING")]
    if debug_mode:
        print("Compile in debug mode")
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["cxx"].append("-O0")
        extra_compile_args["cxx"] += ["-UNDEBUG", "-O0", "-DDEBUG"]
        if "nvcc" in extra_compile_args:
            # we have to remove "-OX" and "-g" flag if exists and append
            nvcc_flags = extra_compile_args["nvcc"]
            extra_compile_args["nvcc"] = [
                f for f in nvcc_flags if not ("-O" in f or "-g" in f)
            ]
            extra_compile_args["nvcc"].append("-O0")
            extra_compile_args["nvcc"].append("-g")
    else:
        extra_compile_args["cxx"] += [("-DNDEBUG"),
                                      "-O3", "-finline-functions"]
        if "nvcc" in extra_compile_args:
            extra_compile_args["cxx"] += [("-DNDEBUG"), "-O3"]
    extra_compile_args["cxx"] += ["-fPIC"]  # +["-fvisibility=hidden"]

    extra_link_args += ["-Wl,-Bsymbolic-functions"]

    not_catch_subthread_exception = os.getenv("NCATCH_SUB", "0") == "1"
    if not_catch_subthread_exception:
        extra_compile_args["cxx"] += [("-DNCATCH_SUB")]
    else:
        extra_compile_args["cxx"] += [("-DCATCH_SUB")]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    image_path = os.path.join(extensions_dir, "backend", "image")

    image_src = glob.glob(os.path.join(image_path, "*.cpp")) + glob.glob(
        os.path.join(image_path, "cuda", "*.cpp")
    )

    image_src += sources
    opencv_includes = []
    
    if WITH_OPENCV:
        defualt_opencv_include = "/usr/local/include/opencv4/"
        if not os.path.exists(defualt_opencv_include):
            defualt_opencv_include = "/usr/include/"

        opencv_include = os.getenv("OPENCV_INCLUDE", defualt_opencv_include)
        if not os.path.exists(os.path.join(opencv_include, "opencv2")):
            raise RuntimeError(
                f"can not find opencv include path. set it by OPENCV_INCLUDE env."
            )
        opencv_includes.append(opencv_include)

        default_opencv_lib_dir = "/usr/local/lib/"
        if not os.path.exists(
            os.path.join(default_opencv_lib_dir, "libopencv_core.so")
        ):
            default_opencv_lib_dir = "/usr/lib/x86_64-linux-gnu/"
        opencv_lib_dir = os.getenv("OPENCV_LIB_DIR", default_opencv_lib_dir)
        if sys.executable.find("conda") != -1:
            extra_link_args += ["-static-libstdc++"]

        core_lib = os.path.join(opencv_lib_dir, "libopencv_core.so")
        if not os.path.exists(core_lib):
            raise RuntimeError(
                f"can not find opencv lib path. set it by OPENCV_LIB_DIR env."
            )
        thirdpart_lib_dirs.append(opencv_lib_dir)

    
    third_includes += [
        os.path.join(extensions_dir, "../..", "thirdparty/dep_sort"),
        os.path.join(extensions_dir, "../..", "thirdparty/digraph/dglib"),
        os.path.join(extensions_dir, "../..", "thirdparty/toml/"),
        os.path.join(extensions_dir, "../..", "thirdparty/AES/src"),
        os.path.join(extensions_dir, "../..", "thirdparty/spdlog/include/"),
    ]

    include_dirs = (
        opencv_includes
        + ["/usr/local/include/", extensions_dir]
        + third_includes
        + [
            os.path.join(extensions_dir, x, "include")
            for x in os.listdir(extensions_dir)
        ]
        + [os.path.join(extensions_dir, x, "src")
           for x in os.listdir(extensions_dir)]
        + [
            os.path.join(extensions_dir, x, "src_cuda")
            for x in os.listdir(extensions_dir)
        ]
    )

    include_dirs += [os.path.join(extensions_dir, "thirdpart")]

    include_dirs = [x for x in include_dirs if os.path.isdir(x)]
    # 添加torch的头文件路径，以便使用version.h. 注意，从pytorch获取其安装路径
    torch_include = os.path.join(os.path.dirname(torch.__file__), "include")
    include_dirs += [torch_include]

    # Image reading extension
    image_macros = []
    image_include = [extensions_dir]
    image_library = []
    image_link_flags = []

    if sys.platform == "win32":
        image_macros += [("USE_PYTHON", None)]

    nvjpeg_found = (
        extension is CUDAExtension
        and CUDA_HOME is not None
        and os.path.exists(os.path.join(CUDA_HOME, "include", "nvjpeg.h"))
    )

    print(f"NVJPEG found: {nvjpeg_found}")
    image_macros += [("NVJPEG_FOUND", str(int(nvjpeg_found)))]

    if use_encrypt:
        extra_lib = ["ssl", "crypto"]
    else:
        extra_lib = []
    if int(build_cuda_version[0]) >= 11:
        link_fs = []
    else:
        link_fs = ["-lstdc++fs"]
    print("extra_compile_args: ", extra_compile_args)

    opencv_libs = []
    if WITH_OPENCV:
        opencv_libs += [
            "opencv_core",
            "opencv_imgcodecs",
            "opencv_imgproc",
            "opencv_highgui",
        ]
    

    if PPLCV_INSTALL:
        if not BUILD_PPLCV:
            assert os.path.exists(PPLCV_INSTALL)

        include_dirs.append(os.path.join(PPLCV_INSTALL, "include"))
        pplcv_lib_dir = os.path.join(PPLCV_INSTALL, "lib")
        thirdpart_lib_dirs.append(pplcv_lib_dir)
        #
        extra_link_args.append(f'-Wl,-rpath={pplcv_lib_dir}')
        extra_link_args.append('-Wl,-Bsymbolic')
        extra_link_args.append(os.path.join(
            pplcv_lib_dir, "libpplcv_static.a"))
        extra_link_args.append(os.path.join(
            pplcv_lib_dir, "libpplcommon_static.a"))

    # ------------------- torchpipe extra extensions ------------------------
    pipe_include = os.environ.get("TORCHPIPE_INCLUDE", None)
    pipe_library = os.environ.get("TORCHPIPE_LIBRARY", None)
    pipe_include = pipe_include.split(
        os.pathsep) if pipe_include is not None else []
    pipe_library = pipe_library.split(
        os.pathsep) if pipe_library is not None else []
    include_dirs += pipe_include
    library_dirs = pipe_library
    include_dirs += [os.path.join(CUDA_HOME, "include")]

    if WITH_TENSORRT:
        thirdpart_libs += ["nvonnxparser", "nvinfer", "nvinfer_plugin"]

    ext_modules = []
    if PPLCV_INSTALL:
        ext_modules.append(CMakeExtension("ppl.cv", "thirdparty/ppl.cv/") )

    ext_modules.append(extension(
        "torchpipe.libipipe",
        sorted(sources),
        include_dirs=include_dirs,
        define_macros=define_macros + image_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args + link_fs,  # "-lstdc++fs"
        library_dirs=["./lib", "/usr/local/lib/"] + thirdpart_lib_dirs,
        libraries=opencv_libs
        + thirdpart_libs
        + ["nppc", "nppig", "nvrtc", "nvjpeg"]
        + extra_lib,
    ))

    return ext_modules


class clean(distutils.command.clean.clean):
    def run(self):
        return
        with open(".gitignore") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    # if ".wheel-process" not in filename:
                    #     continue
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        distutils.command.clean.clean.run(self)


def get_include_dirs():
    import os

    res = []
    dir_name = "./torchpipe/csrc"
    res.append(dir_name)

    for sub_dir in os.listdir(dir_name):
        sub_name = os.path.abspath(os.path.join(dir_name, sub_dir, "include"))
        if os.path.isdir(sub_name):
            res.append(sub_name)

    dir_name = "./thirdparty"
    res.append(os.path.abspath(os.path.join(dir_name, ".")))
    return res


install_files = [os.path.abspath(
    "torchpipe/csrc/core/include/torchpipe/extension.h")]
inc_ = get_include_dirs()
for i in inc_:
    files = os.listdir(i)
    file = [
        os.path.abspath(os.path.join(i, fi))
        for fi in files
        if fi.endswith(".hpp") or fi.endswith(".h")
    ]
    install_files.extend(file)

dir_name = "./thirdparty"

for root, dir_local, names in os.walk(
    os.path.abspath(os.path.join(dir_name, "spdlog/include"))
):
    for name in names:
        install_files.append(os.path.join(root, name))

    
if __name__ == "__main__":
    # print("install_files: \n", install_files)
    print(f"Building wheel {package_name}-{version}")

    write_version_file()

    with open("README.md", encoding="utf-8") as f:
        readme = f.read()

    check_sub_modules()

    setup_requires = ["torch"]
    if PPLCV_INSTALL:
        setup_requires += ["cmake"]
    ext = get_extensions()

    # data_files = []
    # if WITH_CVCUDA:
    #     assert CVCUDA_INSTALL is not None
    #     cvcuda_libdir = os.path.join(CVCUDA_INSTALL, "lib/x86_64-linux-gnu/")
    #     files = ["libcvcuda.so.0.5.0", "libnvcv_types.so.0.5.0",
    #              "libcvcuda.so.0", "libnvcv_types.so.0",
    #              "libcvcuda.so", "libnvcv_types.so"]
    #     data_files = [os.path.join(cvcuda_libdir, x) for x in files]   
    #     print(data_files)
    setup(
        # Metadata
        name=package_name,
        version=version,
        author="torchpipe Team",
        author_email="",
        url="",
        description="pipeline parallelism inference library with PyTorch frontend",
        long_description=readme,
        license="Apache License 2.0",
        # Package info
        packages=find_packages(exclude=("test",)),
        package_data={
            "torchpipe": install_files,
        },
        zip_safe=False,
        install_requires=requirements,
        setup_requires=setup_requires,
        extras_require={
            # "scipy": ["scipy"],
        },
        ext_modules=ext,
        python_requires=">=3.7,<=3.12",
        cmdclass={
            "build_ext": CMakeBuild,
            "clean": clean,
        },
    )
