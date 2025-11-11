from __future__ import annotations

import os
import importlib

DIR = os.path.abspath(os.path.dirname(__file__))

def get_C_path() -> str:

    omniback_dir = os.path.join(DIR, "../")
    ext_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    omniback_C_so = (os.path.join(omniback_dir, "omniback", f'_C{ext_suffix}'))
    return omniback_C_so


def get_root() -> str:
    return os.path.join(DIR, "../")


def get_includes() -> str:  # noqa: ARG001
    """
    Return the path to the omniback include directories.
    """
    in_directories = []
    installed_path = os.path.join(DIR, "include")
    if not os.path.exists(installed_path):
        installed_path = os.path.join(DIR, "../cpp/")
        assert os.path.exists(installed_path), installed_path
    re = [installed_path,  os.path.join(
        DIR, "./"),  os.path.join(DIR, "../third_party/spdlog/include/")]
    re += [os.path.join(DIR, "../")]
    return re
    source_path = os.path.join(os.path.dirname(DIR), "include")
    in_directories.append(installed_path if os.path.exists(
        installed_path) else source_path)

    curr = os.path.join(DIR, "../")
    assert os.path.exists(os.path.join(curr, "omniback", "csrc")), curr
    in_directories.append(curr)

    third = os.path.join(curr, "./third_party/spdlog/include/")
    assert os.path.exists(third)
    in_directories.append(third)
    return in_directories


def get_library_dir() -> str:  # noqa: ARG001
    lib_path = os.path.join(DIR, ".")
    so_path = os.path.join(lib_path, "libomniback.so")
    if not os.path.exists(so_path):
        lib_path = os.path.join(DIR, "../.setuptools-cmake-build/omniback")
        so_path = os.path.join(lib_path, "libomniback.so")
    if not os.path.exists(so_path):
        raise RuntimeError(f"{so_path} not exist")

    return lib_path


def get_cmake_dir() -> str:
    """
    Return the path to the omniback CMake module directory.
    """
    cmake_installed_path = os.path.join(DIR, "share", "cmake", "omniback")
    if os.path.exists(cmake_installed_path):
        return cmake_installed_path

    msg = "omniback not installed, installation required to access the CMake files"
    raise ImportError(msg)


def get_pkgconfig_dir() -> str:
    """
    Return the path to the omniback pkgconfig directory.
    """
    pkgconfig_installed_path = os.path.join(DIR, "share", "pkgconfig")
    if os.path.exists(pkgconfig_installed_path):
        return pkgconfig_installed_path

    msg = "omniback not installed, installation required to access the pkgconfig files"
    raise ImportError(msg)
