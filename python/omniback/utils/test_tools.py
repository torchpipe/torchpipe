# Throughput and latency test tools

version = "20230104.0"

"""
Performance and latency testing tools for omniback.

Update history:
- 0.0.1 2022-03-15: Initial version
- 0.1.1 2022-03-16: Add multiple requests per call
- 0.1.2 2022-03-18: Fix avg time calculation
- 0.1.3 2022-03-18: Random data for each request
- 0.1.4 2022-03-18: Documentation and typo fixes
- 0.1.5 2022-03-24: Add MEAN metric
- 0.1.6 2022-05-26: Add CPU and memory monitoring
- 0.1.7 2022-06-17: Custom pid support
- 0.1.8 2022-07-28: Direct jpg binary reading
- 2022-09-08: Add callback support
- 2023-01-04: Add PRELOAD_TYPE option
"""

import torch
from timeit import default_timer as timer
import cv2
import sys
import random
import os
import threading
import numpy as np


max_pic_loaded = 999
infer_prob_print = True

PRELOAD_TYPE = os.environ.get("PRELOAD_TYPE", "auto")


class TestParams:
    def __init__(self):
        self.random_choice = True


def pre_resize(im, max_size=640.0):
    h, w, _ = im.shape
    if max(h, w) <= max_size:
        return im
    ratio = max_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return im


def preload_images(file_dir, recursive=True, ext=[".jpg", ".JPG", ".jpeg", ".JPEG"]):
    """Preload image files from directory."""
    if not os.path.exists(file_dir):
        raise RuntimeError(file_dir + " not exists")

    list_images = []
    if recursive:
        for root, folders, filenames in os.walk(file_dir):
            for filename in filenames:
                if os.path.splitext(filename)[-1] in ext:
                    list_images.append(os.path.join(root, filename))
    else:
        list_images = [
            x for x in os.listdir(file_dir) if os.path.splitext(x)[-1] in ext
        ]
        list_images = [os.path.join(file_dir, x) for x in list_images]

    return list_images


def preload(file_dir, recursive=True, ext=[".jpg", ".JPG", ".jpeg", ".JPEG"], max_pic=None):
    """Preload images with optional limit."""
    list_images = preload_images(file_dir, recursive, ext)

    if len(list_images) == 0:
        raise RuntimeError("find no valid files. ext = " + ext)

    if max_pic is not None and max_pic > 0:
        list_images = list_images[:max_pic]

    result = []
    for file_path in list_images:
        with open(file_path, "rb") as f:
            result.append((file_path, f.read()))

    return result


def preload_with_decoding(file_dir, recursive=True, ext=[".jpg", ".JPG", ".jpeg", ".JPEG"], max_pic=None):
    """Preload and decode images."""
    list_images = preload_images(file_dir, recursive, ext)

    if len(list_images) == 0:
        raise RuntimeError("find no valid files. ext = " + ext)

    if max_pic is not None and max_pic > 0:
        list_images = list_images[:max_pic]

    result = []
    for file_path in list_images:
        img = cv2.imread(file_path)
        if img is not None:
            result.append(img)

    return result


class GpuInfo:
    """GPU monitoring using pynvml."""

    def __init__(self, pid=0):
        import pynvml

        self.pynvml = pynvml
        pynvml.nvmlInit()

        self.need_record_index = -1
        gpuDeviceCount = pynvml.nvmlDeviceGetCount()

        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if len(CUDA_VISIBLE_DEVICES) == 1:
            self.need_record_index = int(CUDA_VISIBLE_DEVICES[0])
        else:
            raise RuntimeError("CUDA_VISIBLE_DEVICES: only support single gpu")

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.need_record_index)

    def get_pid_info(self):
        return self.pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu

    def __del__(self):
        pass


def test_from_ids(forward_function, ids, request_batch=1):
    """Test with pre-defined IDs."""
    assert isinstance(forward_function, list)
    assert len(ids) > 0

    data = ids * (max_pic_loaded // len(ids) + 1)
    data = data[:max_pic_loaded]

    instance_threads = []
    for i in range(len(forward_function)):
        instance_threads.append(
            {
                "forward": forward_function[i],
                "id": data[i * len(data) // len(forward_function) :],
            }
        )

    warm_up_num = 2
    for _ in range(warm_up_num):
        for thread in instance_threads:
            thread["forward"](thread["id"][:request_batch])

    final_result = []
    for thread in instance_threads:
        result = []
        for id_ in thread["id"]:
            start = timer()
            thread["forward"]([id_])
            result.append(timer() - start)
        final_result.append(result)

    return final_result


def test_from_raw_file(forward_function, file_dir, num_clients=10, request_batch=1, total_number=10000, recursive=True):
    """Test performance with raw file data."""
    data = preload(file_dir=file_dir, recursive=recursive)

    print(f"file_dir = {file_dir}, num_clients = {num_clients}, request_batch = {request_batch}, total_number = {total_number}")
    assert len(data) > 0
    if total_number == 0:
        total_number = len(data)

    assert total_number > 0

    if isinstance(forward_function, list):
        assert len(forward_function) == num_clients
    else:
        forward_function = [forward_function] * num_clients

    total_time = 0
    list_latency = []

    import math
    repeats = math.ceil(total_number / len(data))
    data = (data * repeats)[:total_number]

    from concurrent.futures import ThreadPoolExecutor

    def run_batch(forward_func, batch_data):
        start = timer()
        forward_func(batch_data)
        return timer() - start

    with ThreadPoolExecutor(max_workers=num_clients) as executor:
        futures = []
        for i in range(num_clients):
            batch_data = data[i::num_clients]
            futures.append(executor.submit(run_batch, forward_function[i], batch_data))

        for f in futures:
            list_latency.extend(f.result())

    list_latency.sort()
    length = len(list_latency)

    tp50 = round(list_latency[length // 2] * 1000, 2)
    tp90 = round(list_latency[int(0.9 * length)] * 1000, 2)
    tp99 = round(list_latency[int(0.99 * length)] * 1000, 2)
    mean = round(sum(list_latency) / len(list_latency) * 1000, 2)

    print("------------------------------Summary------------------------------")
    print(f"num_clients:: {num_clients}")
    print(f"total_number::   {total_number}")
    print(f"latency::        TP50: {tp50}   TP90: {tp90}   TP99:  {tp99}   avg: {mean} ms")
    print("-------------------------------------------------------------------")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir", help="Image directory")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--request_batch", type=int, default=1)
    parser.add_argument("--total_number", type=int, default=10000)
    parser.add_argument("--recursive", type=bool, default=True)
    args = parser.parse_args()

    def dummy_forward(batch_data):
        pass

    test_from_raw_file(dummy_forward, args.file_dir, args.num_clients, args.request_batch, args.total_number, args.recursive)
