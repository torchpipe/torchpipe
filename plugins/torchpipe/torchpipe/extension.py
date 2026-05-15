"""Extension loading utilities for TorchPipe."""

from __future__ import annotations

import ctypes
import importlib
import logging
import os

import torch

def _get_extension_path(lib_name: str) -> str:
    """Get the path to a native extension library.

    Args:
        lib_name: Name of the library to find

    Returns:
        Path to the library file

    Raises:
        ImportError: If library is not found
        RuntimeError: If running on Windows (not supported)
    """
    if os.name == "nt":
        raise RuntimeError("Windows is not supported")

    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError(f"Extension {lib_name} not found")

    return ext_specs.origin

def _check_cuda_version() -> bool:
    """Verify CUDA version compatibility.

    Returns:
        True if versions are compatible
    """
    # TODO: Implement actual CUDA version checking
    return True

def _load_library(lib_name: str) -> None:
    """Load a native extension library.

    Args:
        lib_name: Name of the library to load
    """
    lib_path = _get_extension_path(lib_name)
    logging.info("Loading %s", lib_path)
    torch.ops.load_library(lib_path)


# Verify CUDA compatibility on module load
_check_cuda_version()
