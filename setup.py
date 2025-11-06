# Copyright 2021-2025 NetEase.
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

# from distutils.spawn import find_executable
import shutil
# from distutils import sysconfig, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

# from logging import log
import logging as log

from contextlib import contextmanager
import os
import shlex
import subprocess
import sys
import platform
import multiprocessing
import glob
import shutil


################################################################################
# Packages
################################################################################
required_setup_deps = ["cmake", "ninja", "pybind11", "setuptools_scm"]


def is_package_installed(package_name):
    try:
        import importlib
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package_name])


# Check and install missing dependencies
for dep in required_setup_deps:
    if not is_package_installed(dep):
        print(f"Installing missing setup dependency: {dep}")
        install_package(dep)


TOP_DIR = os.path.realpath(os.path.dirname(__file__))

CMAKE_BUILD_DIR = os.path.join(TOP_DIR, ".setuptools-cmake-build")
CMAKE_LIBRARY_OUTPUT_DIRECTORY = os.path.join(CMAKE_BUILD_DIR, ("hami/"))

_debug = int(os.environ.get("DEBUG", 0))
CMAKE_BUILD_TYPE = "Debug" if _debug else "Release"

WINDOWS = os.name == "nt"

CMAKE = shutil.which("cmake3") or shutil.which("cmake")
MAKE = shutil.which("make")

BUILD_INFO = {}
################################################################################
# Global variables for controlling the build variant
################################################################################


def use_cxx11_abi():
    abi11 = os.environ.get("USE_CXX11_ABI", None)
    if abi11 is None:
        command = [
            sys.executable,
            '-c',
            'import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)'
        ]

        try:
            result = subprocess.check_output(command, text=True)
            abi11 = result.strip() == "True"
        except subprocess.CalledProcessError as e:
            print(f"Error checking ABI: {e}")
            abi11 = False
    else:
        if abi11 in ['False', '0']:
            abi11 = False
        else:
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
        raise RuntimeError(
            "Can only cd to absolute path, got: {}".format(path))
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

            ninja = shutil.which('ninja')
            assert ninja
            if ninja:
                # pass in the ninja build path
                cmake_args = [CMAKE, "-G", "Ninja",
                              f"-DCMAKE_MAKE_PROGRAM={ninja}"]
            else:
                cmake_args = [CMAKE]

            # configure
            cmake_args += [
                "-DPYTHON_EXECUTABLE={}".format(sys.executable),
                "-DPython3_EXECUTABLE={}".format(sys.executable),
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                "-DPY_VERSION={}".format(
                    str(sys.version_info[0]) + "." + str(sys.version_info[1])
                ),
                "-DCMAKE_C_COMPILER={}".format(shutil.which("gcc")),
                "-DCMAKE_CXX_COMPILER={}".format(shutil.which("g++")),
            ]
            cmake_args.append(f"-DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}")
            if _debug:
                cmake_args.append(
                    "-DCMAKE_CXX_FLAGS=-Wall -Werror -Wno-error=sign-compare -O3"),
                cmake_args.append(
                    "-DCMAKE_CXX_FLAGS_DEBUG=-UNDEBUG -O0 -DDEBUG -g")
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

            cmake_args.append(
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
            cmake_args += [
                "-DBUILD_PYBIND=ON",  # build pybind
                "-DUSE_CCACHE=ON",  # use ccache if available
                "-DUSE_MANYLINUX:BOOL=ON",  # use manylinux settings
            ]

            if use_cxx11_abi():
                cmake_args += ["-DUSE_CXX11_ABI=ON"]
            else:
                cmake_args += ["-DUSE_CXX11_ABI=OFF"]
            log.info(f"USE_CXX11_ABI: {use_cxx11_abi()}")

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
            # cmake_args.extend(["--target", "_C"])

            subprocess.check_call(build_args, env=env)
            # import pdb; pdb.set_trace()

            BUILD_INFO["USE_CXX11_ABI"] = use_cxx11_abi()

            import pybind11
            BUILD_INFO["PYBIND11_VERSION"] = pybind11.__version__
            BUILD_INFO["CMAKE_BUILD_TYPE"] = CMAKE_BUILD_TYPE
            BUILD_INFO["PYTHON_VERSION"] = f"{sys.version_info.major}.{sys.version_info.minor}"


class build_ext(setuptools.command.build_ext.build_ext):
    def copy_header_files(self, src_dir, dst_dir, extensions=('.h', '.hpp')):
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} does not exist")
            return

        os.makedirs(dst_dir, exist_ok=True)

        for root, _, files in os.walk(src_dir):
            rel_path = os.path.relpath(root, src_dir)
            dest_path = os.path.join(dst_dir, rel_path)

            os.makedirs(dest_path, exist_ok=True)

            for file in files:
                if file.lower().endswith(extensions):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dest_path, file)
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")

    def run(self):

        self.run_command("cmake_build")

        self.copy_header_files(
            src_dir="cpp/hami",
            dst_dir=os.path.join(self.build_lib, "hami/include/hami"),
            extensions=('.h', '.hpp')
        )

        # 复制 spdlog 头文件
        self.copy_header_files(
            src_dir="third_party/spdlog/include/spdlog",
            dst_dir=os.path.join(self.build_lib, "hami/include/spdlog"),
            extensions=('.h', '.hpp')
        )

        return super().run()

    def build_extensions(self):

        build_lib = self.build_lib
        extension_dst_dir = os.path.join(build_lib, "hami/")

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


def get_install_files_old():
    incs = []

    all_top = ['./third_party/spdlog/include/', 'cpp/']
    for item in all_top:
        for root, dirs, files in os.walk(os.path.join(TOP_DIR, item)):
            for file in files:
                if file.endswith(".h") or file.endswith(".hpp"):
                    incs.append(os.path.join(root, file))
    print(
        f"Found {len(incs)} header files to install. incs[0]= {incs[0] if incs else 'None'}")
    return incs


setuptools.setup(
    name="hami-core",

    packages=[
        "hami",
        'hami.csrc',
        "hami.utils",  # 显式声明
        "hami._C"     # 如果 _C 也是包
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    package_data={
        "hami": [
            'csrc/',
            "*.so",
            "csrc/*.hpp",  # 明确包含 .hpp 文件
            "csrc/*.h",    # 明确包含 .h 文件
            "_C*.so",
            "_build_info.py",
        ]
    },
    setup_requires=required_setup_deps
)
