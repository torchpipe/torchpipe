# Copyright 2021-2024 NetEase.
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


import os
import platform
import tempfile

IS_WINDOWS = platform.system() == "Windows"
IS_DARWIN = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

from torchpipe import WITH_CUDA

def make_relative_rpath_args(path):
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


torchpipe_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")

include_dir = os.path.join(torchpipe_dir, "../", "thirdparty/any")
# print(include_dir)
if not os.path.exists(os.path.join(include_dir, "any.hpp")):
    include_dir = "/usr/local/torchpipe/thirdparty/any"

include_dirs = [
    torchpipe_dir,
    os.path.join(torchpipe_dir, "csrc", "./core/include"),
    os.path.join(torchpipe_dir, "csrc", "./backend/include"),
    os.path.join(torchpipe_dir, "csrc", "./schedule/include"),
    os.path.join(torchpipe_dir, "csrc", "./pipeline/include"),
    os.path.join(torchpipe_dir, "csrc"),
    os.path.join(torchpipe_dir, "../thirdparty/spdlog/include/"),
    "/usr/local/include/",
    "/usr/local/include/opencv4/"
] + [include_dir]

DEFAULT_REBUILD_IF_EXIST = os.environ.get("DEBUG", "0") == "1"

lib_dir = ""
lib_path = ""
for root, dirs, files in os.walk(os.path.join(torchpipe_dir, "..")):
    for name in files:
        if name == "libipipe.so":
            lib_path = os.path.join(root, name)
            lib_dir = root


def _import_module_from_library(module_name, path, is_python_module):
    import torch
    import sys
    import importlib

    IS_WINDOWS = sys.platform == "win32"
    LIB_EXT = ".pyd" if IS_WINDOWS else ".so"
    filepath = os.path.join(path, f"{module_name}{LIB_EXT}")
    if is_python_module:
        # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        assert isinstance(spec.loader, importlib.abc.Loader)
        spec.loader.exec_module(module)
        return module
    else:
        torch.ops.load_library(filepath)

def load_filter(
    name="",
    sources = "",
    sources_header="",
    rebuild_if_exist=DEFAULT_REBUILD_IF_EXIST,
    is_python_module=False,
    extra_include_paths=[],
    extra_ldflags=[],
    with_cuda=WITH_CUDA,
    *args,
    **kwargs,
):
    if len(name) == 0:
        import random
        name = str(random.random())
    cls_name = name.replace(".","")
    # 通过tempfile获得临时文件夹
    if True:
        # 将源文件写入临时文件夹
        file_context_start = f'''
        #include <torch/extension.h>
        #include "filter.hpp"
        #include "reflect.h"
        {sources_header};
        //using Filter = ipipe::Filter;
        //using status = Filter::status;
        namespace ipipe{{
        namespace {{
        class Filter{cls_name} : public Filter {{
            public:
            {sources}
        }};
        }}
        IPIPE_REGISTER(Filter, Filter{cls_name}, "{name}");
        }}
        
        '''
        from torch.utils.cpp_extension import _get_build_directory

        target_exten_dir = _get_build_directory(name, False)
        tmpdir = target_exten_dir

        with open(os.path.join(tmpdir, f"{name}.cpp"), "w") as f:
            f.write(file_context_start)
            print(f"write source file to {tmpdir}/{name}.cpp")
            print(file_context_start)
        # 调用load函数编译
        load(
            name=name,
            sources=[os.path.join(tmpdir, f"{name}.cpp")],
            rebuild_if_exist=rebuild_if_exist,
            is_python_module=is_python_module,
            extra_include_paths=extra_include_paths ,
            extra_ldflags=extra_ldflags,
            with_cuda=with_cuda,
            *args,
            **kwargs,
        )
        # 调用import_module_from_library函数导入
        return _import_module_from_library(name, tmpdir, is_python_module)
    

def load_backend(
    name="",
    sources = "",
    sources_header="",
    rebuild_if_exist=DEFAULT_REBUILD_IF_EXIST,
    is_python_module=False,
    extra_include_paths=[],
    extra_ldflags=[],
    with_cuda=True,
    *args,
    **kwargs,
):
    if len(name) == 0:
        import random
        name = str(random.random())
    cls_name = name.replace(".","")
    # 通过tempfile获得临时文件夹
    if True:
        # 将源文件写入临时文件夹
        file_context_start = f'''
        #include <torch/extension.h>
        #include "Backend.hpp"
        #include "reflect.h"
        {sources_header};
        //using Backend = ipipe::Backend;

        namespace ipipe{{
        namespace {{
        class Backend{cls_name} : public SingleBackend {{
            public:
            {sources}
        }};
        }}
        IPIPE_REGISTER(Backend, Backend{cls_name}, "{name}");
        }}
        
        '''
        from torch.utils.cpp_extension import _get_build_directory

        target_exten_dir = _get_build_directory(name, False)
        tmpdir = target_exten_dir

        with open(os.path.join(tmpdir, f"{name}.cpp"), "w") as f:
            f.write(file_context_start)
            print(f"write source file to {tmpdir}/{name}.cpp")
            print(file_context_start)
        # 调用load函数编译
        load(
            name=name,
            sources=[os.path.join(tmpdir, f"{name}.cpp")],
            rebuild_if_exist=rebuild_if_exist,
            is_python_module=is_python_module,
            extra_include_paths=extra_include_paths ,
            extra_ldflags=extra_ldflags,
            with_cuda=with_cuda,
            *args,
            **kwargs,
        )
        # 调用import_module_from_library函数导入
        return _import_module_from_library(name, tmpdir, is_python_module)
    
    # call load to compile filter
    

def load(
    name="",
    sources=[],
    rebuild_if_exist=DEFAULT_REBUILD_IF_EXIST,
    is_python_module=False,
    extra_include_paths=[],
    extra_ldflags=[],
    with_cuda=WITH_CUDA,
    *args,
    **kwargs,
):
    """make use of torch.utils.cpp_extension to compile related source files.

    :param sources: list of files' path
    :type sources: List
    :param name: name of library
    :type name: str, optional
    :param extra_include_paths: include paths, defaults to []
    :type extra_include_paths: list, optional
    :param extra_ldflags: extra c++ ldflags, defaults to []
    :type extra_ldflags: list, optional
    :param rebuild_if_exist: rebuild if the generated library exists, defaults to False
    :type rebuild_if_exist: bool, optional
    :param is_python_module: parameter for torch.utils.cpp_extension.load. defaults to False
    :type is_python_module: bool, optional
    :return: imported library
    """
    import sys
    import torch

    if name == "" and len(sources) >= 1:
        name = sources[0].split(".")[0]

    from torch.utils.cpp_extension import _get_build_directory

    target_exten_dir = _get_build_directory(name, False)
    target_lib_path = os.path.join(target_exten_dir, f"{name}.so")
    if not rebuild_if_exist:
        if os.path.exists(target_lib_path):
            print(
                f"Load plugin without compiling: {target_lib_path} . Delete it if you want to recompile source files."
            )
            return _import_module_from_library(name, target_exten_dir, is_python_module)
            from ctypes import cdll

            print(
                f"Load plugin without compiling: {target_lib_path} . Delete it if you want to recompile source files."
            )
            library = cdll.LoadLibrary(target_lib_path)

            return library
    print(f"\nStart to generate {name}.so\n")

    _DEBUG = False
    if "DEBUG" in os.environ:
        assert os.environ["DEBUG"] in ["0", "1"]
        if os.environ["DEBUG"] == "1":
            _DEBUG = True

    _DEBUG_LEVEL = 0
    extra_compile_args = ["-DPYBIND", "-std=c++17"]  # + ["-std=c++14"] '-fopenmp',
    if _DEBUG:
        extra_compile_args += ["-g3", "-O0", "-DDEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
    else:
        extra_compile_args += ["-DNDEBUG", "-O2", "-finline-functions"]
    print("extra_compile_args: ", extra_compile_args)
    from torch.utils.cpp_extension import load, load_inline

    # file_datas = []
    # for i in sources:
    #     with open(i,  encoding='utf-8') as f:
    #         file_data = f.read()
    #     file_data = file_data.encode('ascii', 'ignore').decode('ascii')  # 去除中文
    #     file_datas.append(file_data)

    extra_ldflags_rpath = make_relative_rpath_args(lib_dir)
    try:
        return load(
            name=name,
            sources=sources,
            extra_cflags=extra_compile_args,
            extra_include_paths=[include_dir] + extra_include_paths + include_dirs,
            extra_ldflags=[f"-L{lib_dir}", "-lipipe", "-L/usr/lib/x86_64-linux-gnu/"]
            + extra_ldflags
            + extra_ldflags_rpath,
            verbose=True,
            is_python_module=is_python_module,
            with_cuda=with_cuda,
        )

    except Exception as e:
        # print(f"exception {e} catched, retrying ")
        # print(f"exception {e} catched ")
        raise e

        return load_inline(
            name=name,
            cpp_sources=file_datas,
            extra_cflags=extra_compile_args,
            extra_include_paths=[include_dir] + extra_include_paths + include_dirs,
            extra_ldflags=[f"-L{lib_dir}", "-lipipe"]
            + extra_ldflags
            + extra_ldflags_rpath,
            verbose=True,
            is_python_module=is_python_module,
            with_cuda=with_cuda,
        )
