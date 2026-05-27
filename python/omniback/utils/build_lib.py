# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Build Torch Addon."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path
# from ._system_path import system_include_dirs, system_library_dirs
try:
    import torch
    missing_dtypes = [
        "uint16", "uint32", "uint64",
        "float8_e5m2fnuz", "float8_e4m3fnuz",
        "float8_e4m3fn", "float8_e5m2"
    ]
    for dtype in missing_dtypes:
        if not hasattr(torch, dtype):
            setattr(torch, dtype, None)
except Exception:
    torch = None

try:
    import torch.torch_version
    import torch.utils.cpp_extension
except Exception:
    pass

import logging
logger = logging.getLogger(__name__)  # type: ignore


# Important: to avoid cyclic dependency, we avoid import tvm_ffi names at top level here.

IS_WINDOWS = sys.platform == "win32"
IS_DARWIN = sys.platform == "darwin"


def unique_paths(paths):
    seen = set()
    unique = []
    for p in paths:
        p_str = str(p)
        norm_p = os.path.abspath(p_str)
        if norm_p not in seen:
            seen.add(norm_p)
            unique.append(norm_p)  # Keep normalized path
    return unique

def get_cpp_source(source_dir):
    source_path = []
    if isinstance(source_dir, list):
        for src_dir in source_dir:
            src = get_cpp_source(src_dir)
            source_path += src
    else:
        import glob
        source_dir = os.path.abspath(source_dir)
        for ext in ('*.cpp', '*.cc', '*.cxx'):
            source_path.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))

    return source_path

def _check_pkg_resources() -> None:
    """Ensure pkg_resources is available for torch.utils.cpp_extension.

    torch<2.0 uses pkg_resources (setuptools<81) internally.
    torch>=2.0 no longer depends on it, so skip the check.
    """
    # Only needed on PyTorch < 2.0
    try:
        if torch.__version__ >= torch.torch_version.TorchVersion("2.0"):
            return
    except (AttributeError, ImportError):
        pass  # old torch without torch_version; likely < 2.0

    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        logger.error(
            "pkg_resources (setuptools<81) is required for JIT compilation "
            "with older PyTorch versions (<2.0). Install it with:\n"
            "  pip install 'setuptools<81'"
        )
        raise


def get_torch_include_paths(build_with_cuda: bool) -> Sequence[str]:
    """Get the include paths for building with torch."""
    _check_pkg_resources()
    from torch.utils.cpp_extension import include_paths as _include_paths

    # torch.torch_version may not exist in older PyTorch (e.g. 1.13).
    # Fall back to the cuda= kwarg API when the version check itself fails.
    try:
        if torch.__version__ >= torch.torch_version.TorchVersion("2.6.0"):
            return _include_paths(
                device_type="cuda" if build_with_cuda else "cpu"
            )
    except (AttributeError, ImportError):
        pass

    return _include_paths(cuda=build_with_cuda)  # type: ignore[call-arg]

def get_cache_dir():
    return str(Path(os.environ.get("OMNIBACK_CACHE_DIR",
                                   "~/.cache/omniback/")).expanduser())


def get_cache_lib(name: str, device: str, no_torch: bool):
    suffix = ".dll" if IS_WINDOWS else ".so"
    return os.path.join(get_cache_dir(), get_cache_name(name, device, no_torch))+suffix


def get_lib_name(name: str, device: str, no_torch: bool):
    suffix = ".dll" if IS_WINDOWS else ".so"
    return get_cache_name(name, device, no_torch)+suffix

def get_cache_name(name: str, device: str, no_torch: bool, abiflag: str = ""):
    # resolve library name
    if abiflag == "":
        abiflag = str(int(torch.compiled_with_cxx11_abi()))
    if no_torch:
        return f"{name}-abiflag{abiflag}"
    major, minor = torch.__version__.split(".")[:2]

    cuda_suffix = ""
    if device == "cuda" and hasattr(torch.version, "cuda") and torch.version.cuda:
        cuda_major = torch.version.cuda.split('.')[0]
        device = f"cuda{cuda_major}"

    suffix = ".dll" if IS_WINDOWS else ".so"
    return f"{name}-torch{major}{minor}-{device}-abiflag{abiflag}"

def main() -> None:  # noqa: PLR0912, PLR0915
    """Build the torch extension."""
    os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
    from tvm_ffi.utils.lockfile import FileLock  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Build the torch c dlpack extension. After building, a shared library will be placed in the output directory.",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        required=False,
        help="Directory to store the built extension library. If not provided, a temporary directory will be used.",
    )
    parser.add_argument(
        '--source-dirs',
        type=str,
        nargs='+',
        required=True,
        help='One or more source directories to search for C++ files'
    )
    parser.add_argument(
        '--include-dirs',
        type=str,
        nargs='+',
        required=True,
        help='One or more include directories to search for C++ headers'
    )
    parser.add_argument(
        '--ldflags',
        type=str,
        nargs='+',
        default=[],
        help='One or more include directories to search for C++ headers'
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default=get_cache_dir(),
        help="Directory to store the built extension library. If not specified, the default cache directory of omniback will be used.",
    )

    parser.add_argument(
        "--abiflag",
        type=str,
        required=False,
        default="",
        help="",
    )

    parser.add_argument(
        "--build-with-cuda",
        action="store_true",
        help="Build with CUDA support.",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="Build with no torch support.",
    )
    parser.add_argument(
        "--build-with-rocm",
        action="store_true",
        help="Build with ROCm support.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="The name of the generated library. ",
    )

    args = parser.parse_args()
    if args.build_with_cuda and args.build_with_rocm:
        raise ValueError("Cannot enable both CUDA and ROCm at the same time.")

    # resolve build directory
    if args.build_dir is None:
        build_dir = Path(tempfile.mkdtemp(prefix="omniback-torch"))
    else:
        build_dir = Path(args.build_dir)
    build_dir = build_dir.resolve()
    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)

    if args.build_with_cuda:
        device = "cuda"
    elif args.build_with_rocm:
        device = "rocm"
    else:
        device = "cpu"

    abiflag = args.abiflag
    if abiflag not in ["1", "0"]:
        assert torch is not None, "torch is not installed. Specify --abiflag option."
        # use CXX11 ABI
        if torch.compiled_with_cxx11_abi():
            abiflag = "1"
        else:
            abiflag = "0"

    libname = get_cache_name(args.name, device, args.no_torch, abiflag)

    tmp_libname = libname + ".tmp"

    # create output directory is not exists
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(output_dir / (libname + ".lock"))):
        if (output_dir / libname).exists():
            # already built
            return

        # get the source
        source_dirs = [Path(d).expanduser() for d in args.source_dirs]
        ldflags = [str(d) for d in args.ldflags]
        source_path = get_cpp_source(source_dirs)

        _include_dirs = [str(Path(d).expanduser()) for d in args.include_dirs]
        include_paths = []
        for inc_dir in _include_dirs:
            if " " in inc_dir:
                include_paths += inc_dir.split()
            else:
                include_paths.append(inc_dir)

        # resolve configs
        cflags = []
        # include_paths.append(sysconfig.get_paths()["include"])

        if abiflag == "1":
            cflags.append("-D_GLIBCXX_USE_CXX11_ABI=1")
        else:
            cflags.append("-D_GLIBCXX_USE_CXX11_ABI=0")

        if not args.no_torch:
            # Import directly to avoid torch.utils.cpp_extension attribute
            # access issues on older PyTorch (e.g. 1.13).
            # COMMON_HIP_FLAGS only exists in ROCm-enabled PyTorch builds.
            _check_pkg_resources()
            from torch.utils.cpp_extension import CUDA_HOME, library_paths
            try:
                from torch.utils.cpp_extension import COMMON_HIP_FLAGS
            except ImportError:
                COMMON_HIP_FLAGS = []

            if args.build_with_cuda:
                cflags.append("-DBUILD_WITH_CUDA")
                if CUDA_HOME is None:
                    logger.error("CUDA_HOME not found")
            elif args.build_with_rocm:
                cflags.extend(COMMON_HIP_FLAGS)
                cflags.append("-DBUILD_WITH_ROCM")
            include_paths.extend(get_torch_include_paths(
                args.build_with_cuda or args.build_with_rocm))

            for lib_dir in library_paths():
                if IS_WINDOWS:
                    ldflags.append(f"/LIBPATH:{lib_dir}")
                else:
                    ldflags.extend(["-L", str(lib_dir)])

            import glob
            if CUDA_HOME is not None:
                cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")
                # dirs = glob.glob(os.path.join(CUDA_HOME, "**/*/lib/stubs"))
                # assert len(dirs) == 1
                if IS_WINDOWS:
                    ldflags.append(f"/LIBPATH:{cuda_lib_dir}")
                else:
                    ldflags.extend(["-L", str(cuda_lib_dir)])

            # Add all required PyTorch libraries
            if IS_WINDOWS:
                # On Windows, use .lib format for linking
                ldflags.extend(
                    ["c10.lib", "torch.lib", "torch_cpu.lib"])
                if args.build_with_cuda:
                    ldflags.extend(["torch_cuda.lib", "c10_cuda.lib"])
            else:
                # On Unix/macOS, use -l format for linking
                ldflags.extend(
                    ["-lc10", "-ltorch", "-ltorch_cpu"])
                if args.build_with_cuda:
                    ldflags.extend(["-ltorch_cuda", "-lc10_cuda"])

        from omniback import get_include_dirs
        import omniback as om
        om_lib = om.libinfo.find_libomniback()

        ldflags.append(f"-L{os.path.dirname(om_lib)}")
        om_lib_name = os.path.splitext(os.path.basename(om_lib))[0].strip('lib')
        if om_lib_name.startswith('lib'):
            om_lib_name = om_lib_name[3:]
        ldflags.append(f"-l{om_lib_name}")

        from tvm_ffi.cpp.extension import build

        include_paths += get_include_dirs()

        include_paths = [str(x) for x in include_paths]
        include_paths = unique_paths(include_paths)
        result_lib = build(name=tmp_libname, cpp_files=[str(x) for x in source_path], extra_cflags=cflags,
                           extra_ldflags=ldflags, extra_include_paths=include_paths, build_directory=build_dir)

        # rename the tmp file to final libname
        final_path = os.path.join(output_dir, os.path.basename(result_lib).replace(".tmp.", '.'))
        shutil.move(str(result_lib), final_path)
        print(f'saved to {final_path}')


# core
# python -m omniback.utils.build_lib --source-dirs csrc/torchplugins/ csrc/helper/ --include-dirs=csrc/ --name torchpipe_core

# nvjpeg
# python -m omniback.utils.build_lib --source-dirs csrc/nvjpeg_torch/ --include-dirs=csrc/ --build-with-cuda --ldflags="-lnvjpeg" --name torchpipe_nvjpeg

# opencv
# python -m omniback.utils.build_lib --no-torch --source-dirs csrc/mat_torch/ --include-dirs csrc/ /usr/local/include/opencv4/ --ldflags "-lopencv_core -lopencv_imgproc -lopencv_imgcodecs" --name torchpipe_opencv

if __name__ == "__main__":
    main()
