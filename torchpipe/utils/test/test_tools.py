# Copyright 2021-2023 NetEase.
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

# throughput and lantecy test
from __future__ import annotations

# from curses import flash
import torch
from timeit import default_timer as timer
import cv2
import sys
import random
import os
import threading
import numpy as np

# sys.path.insert(0, os.path.join("..", os.path.dirname(__file__)))
# sys.path.insert(0, os.path.join(".", os.path.dirname(__file__)))

# from ..device_tools import install_package

from typing import List, Union, Callable, Tuple
from .Sampler import Sampler, RandomSampler, preload, SequentialSampler, FileSampler



version = "20230817"

"""! @package test-tools
# update 0.0.1 2022-03-15 整理出基础版本。 by zsy
# update 0.1.1 2022-03-16 增加一次发送多个请求。 by zsy
# update 0.1.2 2022-03-18 在一次发送多个请求的情况下修复avg 时间的计算。 by zsy
# update 0.1.3 2022-03-18 在一次发送多个请求的情况每个数据都随机。 by zsy
# update 0.1.4 2022-03-18 文档， typo. by zsy
# update 0.1.5 2022-03-24 整理文档，格式调整， typo, 增加MEAN. 实际上MEAN～=avg，
#                         但是少了数据选取（choice）和结果打印等时间, 根据avg和MEAN的值的差距推断，这部分影响在千分之三以内；
#                         可以限制最长边大小，需要手动取消# img=pre_resize(img) 的注释  by zsy
# update 0.1.6 2022-05-26 增加输出当前进程cpu使用情况和内存使用情况中位数  by zsy
# update 0.1.7 2022-06-17 可设置pid；pid不存在时提示并退出；cpu利用率过小时，不显示结果（大概率匹配到错误的进程）fix batch_size>1   by zsy wlc
# update 0.1.8 2022-07-28  直接读取jpg binary， 不在预先解码（文件名后缀需要为 ".jpg", '.JPG', '.jpeg', '.JPEG'）
# update 2022-09-08  增加callback， 用于接收和处理结果；此时推理函数返回类型需要是list类型；
#                           当 total_number <=0 时，变为只跑一遍的模式；
#                           图片数量过少时，不再宕机;   
# update 2023-01-04  增加 PRELOAD_TYPE: yes no auto;                        
# update 2023-04-21  重新整理成类，兼容通用自定义场景;                        
# update 2023-06-02  对齐文档中的新的API，兼容广义场景（API仍处于测试阶段）;                        
# update 2023-07-25  更改主要API为test_from_raw_file;                        
# update 2023-08-17  增加测试结果的返回;                        
# update 2023-11-09  增加gpu使用率中位数输出;                        

# ##########################################################################################################################
# 评价服务性的最佳指标是
# 1. 低于一定时延下的最高吞吐
# 2. 实时性
# 线上服务不像移动端， 不太关注实时性

# 时延依赖于并发请求数目， 并发为5时，此时可能有另外5个服务在排队，排队的时间不算做请求时间， 故只有固定并发请求路数（比如固定客户端数目）下， 算时间延迟才准确。 在并发请求路数少时，时间延迟比测算出来的要多。

# 当我们固定并发请求数目时，此时qps和平均时延又是反相关的， 此时的时延和qps都能够比较准确反映性能。

# 另外，第一条接近于满足一定时间延迟要求下， 比较两个服务同时能够响应的并发请求数目（更准确的，是该并发请求数目下的qps， 不过此时如果可以假设两个服务平均时延差不多，则qps可以由并发请求数决定。
#           但实际上这条假设只是近似成立， 尤其当我们设置的是TP90这类阈值）。这是第二种测量性能的方法， 需要不断增加并发数并判断时延是否超过阈值，然后获得最大并发数以及该并发下的qps。比较动态。

# 总结为，测量低于一定时延下的最高吞吐这一个指标主要有两种方法：
# A. 选择一个或者一组合适的固定的并发请求路数， 如10*2或者20*1，跑固定数量的图片， 如10000张， 算时延（或者qps）
# B. 选择一个或者一组合适的时间延迟阈值，测算TP90等时延指标满足此阈值时，服务所能最大支撑的请求数目。
# ##########################################################################################################################


# 建议固定服务端设置，报告以下三种情况的 qps TP50 TP90 TP99：
# – 10并发请求下 每次数据量1
# – 10并发请求下 每次数据量4（或者2）
# – 40（或者20）并发请求下 每次数据量1 (需要修改thrift设置)
# 或者只报告第一种情况

"""
# 最新版本请访问 torchpipe/tool/test_tools.py




from collections import namedtuple


class TestParams:
    def __init__(self, total_number, num_clients) -> None:
        self.lock = threading.Lock()
        self.result = {}
        self.finish_condition = threading.Condition()

        assert total_number >= 0
        self.total_number = total_number
        self.num_clients = num_clients


# TestParams = namedtuple("TestParams", ["lock", "total_number", "num_clients", "finish_condition", "result"])
# LocalResult = namedtuple("LocalResult", ["latency", "start_time", "end_time"])
class LocalResult:
    def __init__(self) -> None:
        self.latency = []


def pre_resize(im, max_size=640.0):
    h, w, _ = im.shape
    if max(h, w) <= max_size:
        return im
    ratio = max_size / max(h, w)

    resized_w = int(ratio * w)
    resized_h = int(ratio * h)
    im = cv2.resize(im, dsize=(resized_w, resized_h))
    return im


class InferThreadData(threading.Thread):
    def __init__(self, index, test_params: TestParams, forward_class: Sampler) -> None:
        threading.Thread.__init__(self, name=str(index))
        self.params = test_params
        self.batch_size = forward_class.batchsize()
        self.index = index
        self.forward_class = forward_class

        self.num_clients = self.params.num_clients

        self.start_index = 0
        self.should_stop = False
        self.update_local_data()
        self.local_result = LocalResult()

    def update_local_data(self):
        with self.params.lock:
            if self.params.total_number <= 0:
                self.should_stop = True
                return

            if self.params.total_number >= self.batch_size:
                self.params.total_number -= self.batch_size
                self.start_index = self.params.total_number
            else:
                self.params.total_number -= self.batch_size
                self.start_index = 0

    def onFinish(self):
        self.local_result.end_time = timer()
        # while True:
        should_wait = False
        with self.params.lock:
            self.params.num_clients -= 1
            if self.params.num_clients != 0:
                should_wait = True

        if should_wait:
            with self.params.finish_condition:
                self.params.finish_condition.wait()
        else:
            with self.params.finish_condition:
                self.params.finish_condition.notify_all()
        with self.params.lock:
            self.params.result[self.index] = self.local_result
            # print("self.onFinish()", self.params.result.keys(), flush=True)

    def run(self):
        self.local_result.start_time = timer()

        try:
            while not self.should_stop:
                self.forward(self.start_index)
                self.update_local_data()
            self.onFinish()
        except Exception as e:
            import os

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print("subthread error: ", repr(e))
            import os

            os._exit(-1)

        # self.forward_class.onFinish()

    def forward(self, start_index):
        start = timer()

        self.forward_class(start_index)

        result_time = timer() - start
        self.local_result.latency.append((result_time, self.batch_size))
        # return num_batch

    def __del__(self):
        pass

    def warmup(self, num):
        for i in range(num):
            self.forward_class(i % (self.params.total_number))


class GpuInfo(object):
    def __init__(self, pid):
        # 初始化
        # nvml_lib = CDLL("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
        import pynvml  #  pip install  py3nvml # nvidia-ml-py3 pynvml

        self.pynvml = pynvml
        pynvml.nvmlInit()

        self.need_record_index = -1  # 需要记录的进程PID

        gpuDeviceCount = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU块数
        i = -1

        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if len(CUDA_VISIBLE_DEVICES) == 1:
            self.need_record_index = int(CUDA_VISIBLE_DEVICES[0])
        else:
            raise RuntimeError("CUDA_VISIBLE_DEVICES: only support single gpu")
        # while self.need_record_index < 0:
        #     i += 1
        #     if i >= gpuDeviceCount:
        #         assert(False, "node gpu not found")
        #         break
        #     print(i, gpuDeviceCount, type(torch.cuda.current_device()))
        #     handle = pynvml.nvmlDeviceGetHandleByIndex(
        #         i
        #     )  # 获取GPU i的handle，后续通过handle来处理
        #     # info = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
        #     ## gpu_memory_total = info.total #GPU i的总显存
        #     # gpu_memory_used = info.used / NUM_EXPAND #转为MB单位
        #     # all_gpu_used.append(gpu_memory_used) #添加进list

        #     ###还可以直接针对pid的gpu消耗进行统计
        #     info_list = pynvml.nvmlDeviceGetComputeRunningProcesses(
        #         handle
        #     )  # 获取所有GPU上正在运行的进程信息
        #     # print(info_list)
        #     # import pdb; pdb.set_trace()
        #     info_list_len = len(info_list)
        #     gpu_memory_used = 0
        #     if info_list_len > 0:  # 0表示没有正在运行的进程
        #         for info_i in info_list:
        #             # print(info_i.pid, pid)
        #             if info_i.pid == pid:  # 如果与需要记录的pid一致
        #                 # gpu_memory_used += info_i.usedGpuMemory / NUM_EXPAND #统计某pid使用的总显存
        #                 self.need_record_index = info_i
        #                 break
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.need_record_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu

    def get_pid_info(self):
        util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        return util

    def get_gpu_device(self):
        deviceCount = self.pynvml.nvmlDeviceGetCount()
        gpu_list = []
        for i in range(deviceCount):
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
            print("GPU", i, ":", self.pynvml.nvmlDeviceGetName(handle))
            gpu_list.append(i)
        return gpu_list

    def get_free_rate(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_rate = int((info.free / info.total) * 100)
        return free_rate

    def get_gpu_info(self, gpu_id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        M = 1024 * 1024
        gpu_info = "id:{}  total:{}M free:{}M  used:{}M free_rate:{}%".format(
            gpu_id,
            info.total / M,
            info.free / M,
            info.used / M,
            self.get_free_rate(gpu_id),
        )
        return gpu_info

    def __del__(self):
        # 最后要关闭管理工具
        pass
        # self.pynvml.nvmlShutdown()


# note 如果待测试函数有返回值，比如cuda上的tensor，有一定概率会copy到cpu并打印出来（初始概率下约打印10次，后续如果打印对象太大，则相应递减概率，但通常对性能影响小于千分之三
class ResourceThread(threading.Thread):
    def __init__(self, pid, result_list, my_event):
        if pid == 0:
            pid = os.getpid()
            print(f"auto set --pid={pid}. Reset it if necessary")

        threading.Thread.__init__(self, name="ResourceThread:" + str(pid))
        import psutil

        # self.p = psutil.Process(pid)
        try:
            self.p = psutil.Process(pid)
        except:
            print(f"pid {pid} not found")
            exit(0)
        print(f"Resource Monitor started: found {self.p.num_threads()} threads")
        # print(psutil.pids())
        # 'username',, 'status'
        for proc in psutil.process_iter(["pid", "exe", "cmdline"]):
            # if (proc.info["name"] in ["top", "cpptools-srv", "bash", "node", "cpptools", "sshd"]):
            if pid != proc.info["pid"]:
                continue
            print(proc.info)
        # import time
        # time.sleep(22)
        self.result_list = result_list
        self.my_event = my_event

        self.gpu = None

        try :
            import pynvml
            self.gpu = GpuInfo(pid)
        except Exception as e:
            print("gpu info not available: ", e)
            self.gpu = None
        # try:
        #     self.gpu = GpuInfo(pid)
        #     self.gpu.get_gpu_device()
        # except Exception as e:
        #     print(" pynvml :", e)

    def get_cpu_mem(self):
        return self.p.cpu_percent(), self.p.memory_percent()

    def run(self):
        import time
        scale = 1
        index = 0
        while not self.my_event.wait(timeout=2):

            index += 1
            cpu_percent = self.p.cpu_percent()
            mem_percent = self.p.memory_percent()
            if cpu_percent > 0 and mem_percent > 0:
                if self.gpu and (index%scale==0):
                    util = self.gpu.get_pid_info()
                    self.result_list.append((cpu_percent, mem_percent, util))
                else:
                    self.result_list.append((cpu_percent, mem_percent, None))


def test(sample: Union[Sampler, List[Sampler]], total_number=10000):
    if isinstance(sample, list):
        num_clients = len(sample)
        assert num_clients > 0
    elif isinstance(sample, Sampler):
        sample = [sample]
        num_clients = 1
    else:
        raise RuntimeError("must be Union[Sampler, List[Sampler]]")
    assert total_number >= 0
    test_params = TestParams(total_number, num_clients)

    instance_threads = [
        InferThreadData(i, test_params, sample[i]) for i in range(len(sample))
    ]

    warm_up_num = 20
    print(f"Warm-up {warm_up_num} times for each client")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=num_clients) as t:
        for thread_ in instance_threads:
            t.submit(thread_.warmup, warm_up_num)

    torch.cuda.synchronize()
    print("Warm-up finished", flush=True)

    resource_result = []
    resource_event = threading.Event()
    resource_thread = ResourceThread(0, resource_result, resource_event)

    resource_thread.daemon = True
    resource_thread.start()

    for i in instance_threads:
        i.daemon = True

    for i in instance_threads:
        i.start()

    for i in instance_threads:
        i.join()
    resource_event.set()

    torch.cuda.synchronize()

    final_result = test_params.result

    all_time = []
    list_latency = []
    for i in range(num_clients):
        all_time.append(final_result[i].start_time)
        all_time.append(final_result[i].end_time)
        for pair_latency in final_result[i].latency:
            list_latency += [pair_latency[0] * 1000] * pair_latency[1]
    all_time.sort()

    total_time = 1 * (all_time[len(all_time) - 1] - all_time[0])
    resource_thread.join()

    # import pdb; pdb.set_trace()
    gpu_resource_result=[]
    try:
        gpu_resource_result = [x for x in list(zip(*resource_result))[2] if x is not None]
    except:
        pass
    
    cpu_resource_result = list(zip(*resource_result))
    
    try:
        resource_result = np.array(cpu_resource_result)[:2,:].astype(np.float32)
        cpu_ = int(10 * np.median(resource_result[0,:])) / 10
        if cpu_ < 0.8 * 100:
            cpu_ = 0
    except:
        cpu_ = "-"

    try:
        gpu_ = "-"
        if gpu_resource_result:
            gpu_ = int(10 * np.median(gpu_resource_result)) / 10
            if gpu_ < 5:
                gpu_ = "-"
    except Exception as e:
        gpu_ = "-"
        print("gpu_ error", e)

    print("resource every 2s:")
    print(resource_result)
    print("")

    if len(list_latency) + test_params.total_number != total_number:
        print("len(list_latency): ", len(list_latency))
        print(f"total_number: {total_number}")
        print("test_params.total_number", test_params.total_number)
        assert False
    list_latency.sort()
    while len(list_latency) < 50:
        list_latency = list_latency * 2
    length = len(list_latency)
    tp50 = round(list_latency[length // 2], 2)
    tp90 = round(list_latency[int(0.9 * length)], 2)
    tp99 = round(list_latency[int(0.99 * length)], 2)
    tp_1 = round(list_latency[-1], 2)
    tp_2 = round(list_latency[-10], 2)
    tp_3 = round(list_latency[-20], 2)
    tp_4 = round(list_latency[-40], 2)
    tp_5 = round(list_latency[-50], 2)
    mean = round(sum(list_latency) / len(list_latency), 2)

    qps = round(total_number / total_time, 2)
    avg = round(1000 * num_clients / qps, 2)
    # avg = 0
    print("------------------------------Summary------------------------------")
    print(f"tool's version:: {version}")
    print(f"num_clients:: {num_clients}")
    # print(f"request batch::  {batch_size}")

    print(f"total_number::   {total_number}")

    print(f"throughput::     qps:  {qps},   [qps:=total_number/total_time]")
    print(f"                 avg:  {avg} ms   [avg:=num_clients/qps]")

    print(f"latency::        TP50: {tp50}   TP90: {tp90}   TP99:  {tp99} ")
    print(
        f"                 avg:  {mean}   -50,-40,-20,-10,-1: {tp_5},{tp_4},{tp_3},{tp_2},{tp_1} ms"
    )
    if cpu_ != "-":
        print(f"cpu::            usage: {cpu_}%")
    if gpu_ != "-":
        print(f"gpu::            usage: {gpu_}%")
    print(
        "-------------------------------------------------------------------\n",
        flush=True,
    )

    # data_zip = list(zip(*data))
    # x.field_names= data_zip[0]
    # x.add_row(data_zip[1])
    # print(data_zip)
    # x.set_style(prettytable.MARKDOWN)
    # print(x, flush=True)
    data = []
    if False:
        # x.field_names = ["Project",  "Value"]
        data.append(["tool's version", version])
        data.append(["num_clients", num_clients])
        data.append(["total_number", total_number])

        data.append(["throughput::qps", qps])
        data.append(["throughput::avg", f"{avg}"])
        data.append(["latency::TP50", f"{tp50}"])
        data.append(["latency::TP90", f"{tp90}"])
        data.append(["latency::TP99", f"{tp99}"])
        data.append(["latency::avg", mean])
        data.append(["-50,-40,-20,-10,-1", f"{tp_5},{tp_4},{tp_3},{tp_2},{tp_1}"])
        try:
            from prettytable import PrettyTable
            import prettytable

            x = PrettyTable()
            x.field_names = ["Project", "Value"]
            for item in data:
                x.add_row(item)
            x.set_style(prettytable.MARKDOWN)
            print("markdown style:\n\n", x, flush=True)
        except:
            print("if you need markdown style result, run: pip install PrettyTable")
    result = {}
    result["tool's version"] = version
    result["num_clients"] = num_clients
    result["total_number"] = total_number

    result["throughput::qps"] = qps
    result["throughput::avg"] = avg
    result["latency::TP50"] = tp50
    result["latency::TP90"] = tp90
    result["latency::TP99"] = tp99
    result["latency::avg"] = mean
    result["-50"] = tp_5
    result["-50"] = tp_4
    result["-20"] = tp_3
    result["-10"] = tp_2
    result["-1"] = tp_1
    result["cpu_usage"] = cpu_
    result["gpu_usage"] = gpu_
    return result


def test_from_raw_file(
    forward_function: Union[
        Callable[[List[tuple[str, bytes]]]], List[Callable[[List[tuple[str, bytes]]]]]
    ],
    file_dir: str,
    num_clients=10,
    batch_size=1,
    total_number=10000,
    num_preload=1000,
    recursive=True,
    ext=[".jpg", ".JPG", ".jpeg", ".JPEG"],
):
    """
    This function is used to test the performance of a function.
    It can be used to test the performance of a function that processes a single image, or a function that processes a batch of images.
    """
    data = preload(
        file_dir=file_dir, recursive=recursive, num_preload=num_preload, ext=ext
    )

    assert len(data) > 0
    if num_preload <= 0:
        total_number = len(data)

    assert total_number > 0

    if isinstance(forward_function, list):
        assert len(forward_function) == num_clients
    else:
        forward_function = [forward_function] * num_clients

    if num_preload > 0:
        forwards = [RandomSampler(data, batch_size) for i in range(num_clients)]
        for i in range(num_clients):
            forwards[i].forward = forward_function[i]
    else:
        data = [x for x, _ in data]
        forwards = [FileSampler(data, batch_size) for i in range(num_clients)]
        for i in range(num_clients):
            forwards[i].handle_data = forward_function[i]

    return test(forwards, total_number)


def test_function(
    forward_function: Union[Callable, List[Callable]],
    num_clients=10,
    batch_size=1,
    total_number=10000,
):
    """
    This function is used to test the performance of a function.
    :param forward_function: a function or a list of functions.
    :param num_clients: number of clients.
    :param batch_size: batch size.
    :param total_number: total number of data.
    :return: None
    """

    class FunctionSampler:
        def __init__(self, function, batchsize) -> None:
            self.function = function
            self.batch_size = batchsize

        def __call__(self, start_index: int) -> None:
            self.function()

        def batchsize(self):
            return self.batch_size

    if isinstance(forward_function, list):
        assert len(forward_function) == num_clients
    else:
        forward_function = [forward_function] * num_clients
    forwards = [
        FunctionSampler(forward_function[i], batch_size) for i in range(num_clients)
    ]
    return test(forwards, total_number)


import torchpipe


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="Port to listen.", default=8095)
    parser.add_argument(
        "--host", type=str, help="Host to run service", default="localhost"
    )
    parser.add_argument(
        "--img_dir", type=str, help=f"img path. 预读取该目录以及子目录下至多1000张图片", default="img/"
    )
    parser.add_argument("--batch_size", type=int, help="单次请求的数据量", default=1)
    parser.add_argument("--num_clients", type=int, help="并发请求数", default=10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    total_number = 10000

    args = parse_arguments(sys.argv[1:])

    class ThriftInfer:
        """wrapper for thrift's python API. You may need to re-implement this class."""

        def __init__(self, host, port, batch_size) -> None:
            """

            :param host: ip
            :type host: str
            :param port: port
            :type port: int
            :param batch_size: size of sended data in batches.
            :type batch_size: int
            """
            from serve import InferenceService
            from serve.ttypes import InferenceParams

            self.InferenceParams = InferenceParams

            from thrift.transport import TSocket
            from thrift.transport import TTransport
            from thrift.protocol import TBinaryProtocol

            self.transport = TSocket.TSocket(host, port)
            self.transport = TTransport.TBufferedTransport(self.transport)
            self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)

            self.client = InferenceService.Client(self.protocol)

            # Connect!
            self.transport.open()
            self.client.ping()
            self.batch_size = batch_size

        def infer(self, data):
            """batch processing

            :param data: batched data
            :type data: List[(str, bytes)]
            :return:
            :rtype: Any
            """
            return self.client.infer_batch([self.InferenceParams(*x) for x in data])

        def __del__(self):
            self.transport.close()

    def test_thrift_from_raw_file(
        img_dir,
        host="localhost",
        port=8095,
        num_clients=10,
        batch_size=1,
        total_number=10000,
    ):
        instances_ = [ThriftInfer(host, port, batch_size) for i in range(num_clients)]

        test_from_raw_file(
            [x.infer for x in instances_],
            img_dir,
            num_clients,
            batch_size,
            total_number,
        )

    test_thrift_from_raw_file(
        args.img_dir,
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        total_number=total_number,
    )
