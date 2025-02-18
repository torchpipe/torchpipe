from __future__ import annotations

import os
import importlib

DIR = os.path.abspath(os.path.dirname(__file__))

def get_C_path() -> str:
    
    hami_dir =  os.path.join(DIR, "../")
    ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]  
    hami_C_so = (os.path.join(hami_dir, "hami", f'_C{ext_suffix}'))
    return hami_C_so
    
def get_root() -> str:
    return os.path.join(DIR, "../")

def get_includes() -> str:  # noqa: ARG001
    """
    Return the path to the hami include directories.
    """
    in_directories = []
    installed_path = os.path.join(DIR, "../cpp")
    source_path = os.path.join(os.path.dirname(DIR), "include")
    in_directories.append(installed_path if os.path.exists(installed_path) else source_path)
    
    curr = os.path.join(DIR, "../")
    assert os.path.exists(os.path.join(curr, "hami", "csrc"))
    in_directories.append(curr)
    
    third = os.path.join(curr, "./third_party/spdlog/include/")
    assert os.path.exists(third)
    in_directories.append(third)
    return in_directories

def get_library_dir() -> str: # noqa: ARG001
    lib_path = os.path.join(DIR, ".")
    so_path = os.path.join(lib_path, "libhami.so")
    if not os.path.exists(so_path):
        lib_path = os.path.join(DIR, "../.setuptools-cmake-build/hami")
        so_path = os.path.join(lib_path, "libhami.so")
    if not os.path.exists(so_path):
        raise RuntimeError(f"{so_path} not exist")
        
    return lib_path

def get_cmake_dir() -> str:
    """
    Return the path to the hami CMake module directory.
    """
    cmake_installed_path = os.path.join(DIR, "share", "cmake", "hami")
    if os.path.exists(cmake_installed_path):
        return cmake_installed_path

    msg = "hami not installed, installation required to access the CMake files"
    raise ImportError(msg)


def get_pkgconfig_dir() -> str:
    """
    Return the path to the hami pkgconfig directory.
    """
    pkgconfig_installed_path = os.path.join(DIR, "share", "pkgconfig")
    if os.path.exists(pkgconfig_installed_path):
        return pkgconfig_installed_path

    msg = "hami not installed, installation required to access the pkgconfig files"
    raise ImportError(msg)