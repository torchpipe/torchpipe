from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cuda.bindings import runtime as rt


def _check(res):
    err = res[0]
    if err != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    if len(res) == 2:
        return res[1]
    return res


def set_device(device_id: int) -> None:
    _check(rt.cudaSetDevice(int(device_id)))


def create_stream() -> int:
    return int(_check(rt.cudaStreamCreate()))


def destroy_stream(stream: int) -> None:
    _check(rt.cudaStreamDestroy(stream))


def stream_synchronize(stream: int) -> None:
    _check(rt.cudaStreamSynchronize(stream))


def malloc(nbytes: int) -> int:
    return int(_check(rt.cudaMalloc(int(nbytes))))


def free(ptr: int) -> None:
    _check(rt.cudaFree(int(ptr)))


def memcpy_htod_async(dst_ptr: int, src: np.ndarray, stream: int) -> None:
    src = np.ascontiguousarray(src)
    _check(
        rt.cudaMemcpyAsync(
            int(dst_ptr),
            src.ctypes.data,
            int(src.nbytes),
            rt.cudaMemcpyKind.cudaMemcpyHostToDevice,
            int(stream),
        )
    )


def memcpy_dtoh_async(dst: np.ndarray, src_ptr: int, stream: int) -> None:
    dst = np.ascontiguousarray(dst)
    _check(
        rt.cudaMemcpyAsync(
            dst.ctypes.data,
            int(src_ptr),
            int(dst.nbytes),
            rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            int(stream),
        )
    )


@dataclass
class DeviceBuffer:
    ptr: int
    nbytes: int

    def free(self) -> None:
        if self.ptr:
            free(self.ptr)
            self.ptr = 0
