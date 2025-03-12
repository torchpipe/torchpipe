import torch

import os
import importlib
import logging
def _get_extension_path(lib_name):

    lib_dir = os.path.dirname(__file__)
    assert os.name != "nt"

    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError

    return ext_specs.origin

def _check_cuda_version():
    """
    Make sure that CUDA versions match between the pytorch install and torchpipe install
    """
    return True
    if not _HAS_OPS:
        return -1
    from torch.version import cuda as torch_version_cuda

    _version = torch.ops.torchvision._cuda_version()
    if _version != -1 and torch_version_cuda is not None:
        tv_version = str(_version)
        if int(tv_version) < 10000:
            tv_major = int(tv_version[0])
            tv_minor = int(tv_version[2])
        else:
            tv_major = int(tv_version[0:2])
            tv_minor = int(tv_version[3])
        t_version = torch_version_cuda.split(".")
        t_major = int(t_version[0])
        t_minor = int(t_version[1])
        if t_major != tv_major:
            raise RuntimeError(
                "Detected that PyTorch and torchvision were compiled with different CUDA major versions. "
                f"PyTorch has CUDA Version={t_major}.{t_minor} and torchvision has "
                f"CUDA Version={tv_major}.{tv_minor}. "
                "Please reinstall the torchvision that matches your PyTorch install."
            )
    return _version

import ctypes
def _load_library(lib_name):
    
    lib_path = _get_extension_path(lib_name)
    # try:
    #     ctypes.CDLL(lib_path)
    # except:
    logging.info(f"Loading {lib_path}")
    torch.ops.load_library(lib_path)


_check_cuda_version()