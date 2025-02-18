# This file referred to github.com/onnx/onnx.git and https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/setup.py

from distutils.spawn import find_executable
from distutils import sysconfig, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

from contextlib import contextmanager
import os
import shlex
import subprocess
import sys
import platform
import multiprocessing
import glob
import shutil

TOP_DIR = os.path.realpath(os.path.dirname(__file__))

CMAKE_BUILD_DIR = os.path.join(TOP_DIR, ".setuptools-cmake-build")
CMAKE_LIBRARY_OUTPUT_DIRECTORY = os.path.join(CMAKE_BUILD_DIR,("hami/"))

_debug = int(os.environ.get("DEBUG", 0))
CMAKE_BUILD_TYPE = "Debug" if _debug else "Release"

WINDOWS = os.name == "nt"

CMAKE = find_executable("cmake3") or find_executable("cmake")
MAKE = find_executable("make")

BUILD_INFO = {}
################################################################################
# Global variables for controlling the build variant
################################################################################

def use_cxx11_abi():
    abi11 = os.environ.get("USE_CXX11_ABI", None)
    if abi11 is None:
        try:
            import torch
            abi11 = torch._C._GLIBCXX_USE_CXX11_ABI
        except ImportError:
            abi11 = False
    else:
        if abi11 in ['False', '0']:
            abi11 = False
        else :
            abi11 = True
    return abi11

################################################################################
# Pre Check
################################################################################

assert CMAKE, 'Could not find "cmake" executable!'

################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError("Can only cd to absolute path, got: {}".format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


################################################################################
# Customized commands
################################################################################


class cmake_build(setuptools.Command):
    """
    Compiles everything when `python setupmnm.py build` is run using cmake.
    Custom args can be passed to cmake by specifying the `CMAKE_ARGS`
    environment variable.
    The number of CPUs used by `make` can be specified by passing `-j<ncpus>`
    to `setup.py build`.  By default all CPUs are used.
    """

    user_options = [
        (str("jobs="), str("j"), str("Specifies the number of jobs to use with make"))
    ]

    built = False

    def initialize_options(self):
        self.jobs = None

    def finalize_options(self):
        if sys.version_info[0] >= 3:
            self.set_undefined_options("build", ("parallel", "jobs"))
        if self.jobs is None and os.getenv("MAX_JOBS") is not None:
            self.jobs = os.getenv("MAX_JOBS")
        self.jobs = multiprocessing.cpu_count() if self.jobs is None else int(self.jobs)

    def run(self):
        os.makedirs(CMAKE_BUILD_DIR, exist_ok=True)

        with cd(CMAKE_BUILD_DIR):
            # configure
            cmake_args = [
                CMAKE,
                "-G",
                "Ninja",
                "-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_python_inc()),
                "-DPYTHON_EXECUTABLE={}".format(sys.executable),
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                "-DPY_EXT_SUFFIX={}".format(
                    sysconfig.get_config_var("EXT_SUFFIX") or ""
                ),
                "-DPY_VERSION={}".format(
                    str(sys.version_info[0]) + "." + str(sys.version_info[1])
                ),
            ]
            cmake_args.append("-DCMAKE_BUILD_TYPE=%s" % CMAKE_BUILD_TYPE)
            if WINDOWS:
                cmake_args.extend(
                    [
                        "-DPY_VERSION={}".format(
                            "{0}.{1}".format(*sys.version_info[:2])
                        ),
                    ]
                )
                if platform.architecture()[0] == "64bit":
                    cmake_args.extend(["-A", "x64", "-T", "host=x64"])
                else:
                    cmake_args.extend(["-A", "Win32", "-T", "host=x86"])
                cmake_args.extend(["-G", "Visual Studio 16 2019"])
            else:
                cmake_args.append(
                    "-DPYTHON_LIBRARY={}".format(
                        sysconfig.get_python_lib(standard_lib=True)
                    )
                )
            if True:
                cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={CMAKE_LIBRARY_OUTPUT_DIRECTORY}") 
                cmake_args += [
                    "-DBUILD_PYBIND=ON", # build pybind
                    f"-DCMAKE_MAKE_PROGRAM={shutil.which('ninja')}",  # pass in the ninja build path
                    "-DUSE_CCACHE=ON",  # use ccache if available
                    "-DUSE_MANYLINUX:BOOL=ON",  # use manylinux settings
                ]
                if use_cxx11_abi():
                    cmake_args += ["-DUSE_CXX11_ABI=ON"]
                else:
                    cmake_args += ["-DUSE_CXX11_ABI=OFF"]
            
            if "CMAKE_ARGS" in os.environ:
                extra_cmake_args = shlex.split(os.environ["CMAKE_ARGS"])
                # prevent crossfire with downstream scripts
                del os.environ["CMAKE_ARGS"]
                log.info("Extra cmake args: {}".format(extra_cmake_args))
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            env = os.environ.copy()
            subprocess.check_call(cmake_args, env=env)

            build_args = [CMAKE, "--build", os.curdir]
            if WINDOWS:
                build_args.extend(["--config", cfg])
                build_args.extend(["--", "/maxcpucount:{}".format(self.jobs)])
            else:
                build_args.extend(["--", "-j", str(self.jobs)])
            cmake_args.extend(["--target", "_C"])
            subprocess.check_call(build_args, env=env)

            BUILD_INFO["USE_CXX11_ABI"] = use_cxx11_abi()
            import pybind11
            BUILD_INFO["PYBIND11_VERSION"] = pybind11.__version__
            BUILD_INFO["CMAKE_BUILD_TYPE"] = CMAKE_BUILD_TYPE
            BUILD_INFO["PYTHON_VERSION"] = f"{sys.version_info.major}.{sys.version_info.minor}"

class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        self.run_command("cmake_build")
        return super().run()

    def build_extensions(self):
        build_lib = self.build_lib
        extension_dst_dir = os.path.join(build_lib, "hami")
        os.makedirs(extension_dst_dir, exist_ok=True)

        for so_file in glob.glob(os.path.join(CMAKE_LIBRARY_OUTPUT_DIRECTORY, "*.so")):
            shutil.copy(so_file, extension_dst_dir)
            log.info("Copying {} to {}".format(so_file, extension_dst_dir))
            
        build_info_path = os.path.join(extension_dst_dir, "_build_info.py")
        with open(build_info_path, "w") as f:
            f.write("# This file is auto-generated during build\n\n")
            for key, value in BUILD_INFO.items():
                if isinstance(value, bool):
                    f.write(f"{key} = {int(value)}\n")
                else:
                    f.write(f"{key} = '{str(value)}'\n")
        
cmdclass = {
    "cmake_build": cmake_build,
    "build_ext": build_ext,
}

################################################################################
# Extensions
################################################################################

ext_modules = [
    setuptools.Extension(name=str("hami._C"), sources=[])
]

setuptools.setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)