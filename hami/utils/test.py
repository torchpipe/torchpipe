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

# throughput and lantecy test
from __future__ import annotations

version = "20250429"

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
# update 2024-04-24  恢复为单文件，并增加 ProcessAdaptor
# update 2024-05-22  使用batch_size取代request_batch参数。但保持兼容性
# update 2024-10-30  fix test_thrift_from_raw_file 读图过多耗时过长的问题
# update 2024-10-30   移除batch_size参数 
# update 2025-04-29   移除 num_preload 参数 

"""


# from curses import flash

from timeit import default_timer as timer

# import cv2
import sys
import random
import os
import threading
import numpy as np

from typing import List, Union, Callable, Tuple, Any
import math

from typing import List, Tuple
import random, os


class Sampler:
    def __init__(self):
        pass

    def __call__(self, start_index: int) -> None:
        raise NotImplementedError

    def batchsize(self):
        return 1


class RandomSampler(Sampler):
    def __init__(self, data_source: List, batch_size=1):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        assert batch_size > 0

        assert 0 < len(data_source)
        for i in range(batch_size):
            if len(data_source) < batch_size:
                data_source.append(data_source[i])

    def __call__(self, start_index: int):
        data = random.sample(self.data_source, self.batch_size)
        self.forward(data)

    def forward(self, data: List):
        raise RuntimeError("Requires users to implement this function")

    def batchsize(self):
        return self.batch_size


class SequentialSampler(Sampler):
    def __init__(self, data: List, batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        assert len(data) >= batch_size

    def __call__(self, start_index: int) -> None:
        data = self.data[start_index : start_index + self.batch_size]
        self.forward(data)

    def batchsize(self):
        return self.batch_size

    def forward(self, data: List):
        raise RuntimeError("Requires users to implement this function")


class LoopSampler(Sampler):
    def __init__(self, data: List, batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        assert len(data) >= batch_size
        self.length = len(data) - batch_size + 1
        for i in range(batch_size):
            self.data.append(data[i])

    def __call__(self, start_index: int) -> None:
        start_index = start_index % (self.length)
        data = self.data[start_index : start_index + self.batch_size]
        self.forward(data)

    def batchsize(self):
        return self.batch_size

    def forward(self, data: List):
        raise RuntimeError("Requires users to implement this function")


class FileSampler(LoopSampler):
    def __init__(self, data: List, batch_size=1):
        super().__init__(data, batch_size)
        self.local_result = {}

    def forward(self, data: List):
        raw_bytes = []
        for file_path in data:
            with open(file_path, "rb") as f:
                raw_bytes.append((file_path, f.read()))
        self.handle_data(raw_bytes)

    def handle_data(self, raw_bytes):
        raise RuntimeError("Requires users to implement this function")

class Identity:
    def __init__(self, request_batch):
        self.request_batch = request_batch
    def __call__(self, data):
        return data
    def batchsize(self):
        return self.request_batch

def preload(
    file_dir, recursive=True, ext=[".jpg", ".JPG", ".jpeg", ".JPEG"]
) -> List[Tuple[str, bytes]]:
    if not os.path.exists(file_dir):
        raise RuntimeError(file_dir + " not exists")

    list_images = []
    result = []
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

    for file_path in list_images:
  
        result.append(file_path)

    if len(result) == 0:
        raise RuntimeError("find no vaild files. ext = " + ext)

    return result


from collections import namedtuple


class TestParams:
    def __init__(self, total_number, num_clients) -> None:
        self.lock = threading.Lock()
        self.result = {}
        self.finish_condition = threading.Condition()

        assert total_number >= 0
        self.total_number = total_number
        self.num_clients = num_clients


class LocalResult:
    def __init__(self) -> None:
        self.latency = []


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
        self.update_local_data() # keep this
        self.local_result = LocalResult()

    def update_local_data(self):
        with self.params.lock:
            if self.params.total_number <= 0:
                self.should_stop = True
                return
            elif self.params.total_number%2000 == 1999:
                print(f"{self.params.total_number}... left", flush=True)

            if self.params.total_number >= self.batch_size:
                self.params.total_number -= self.batch_size
                self.start_index = self.params.total_number
            else:
                self.params.total_number -= self.batch_size
                self.start_index = 0
            # print(f'{self.params.total_number}, {self.batch_size}')

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

        # print(f'self.forward_class={self.forward_class}')
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
        raise NotImplementedError
        return 
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_rate = int((info.free / info.total) * 100)
        return free_rate

    def get_gpu_info(self, gpu_id):
        raise NotImplementedError
        return 
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

        try:
            import pynvml

            self.gpu = GpuInfo(pid)
        except Exception as e:
            print("gpu info not available: ", e)
            self.gpu = None

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
                if self.gpu and (index % scale == 0):
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

    # torch.cuda.synchronize()
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

    gpu_resource_result = []
    try:
        gpu_resource_result = [
            x for x in list(zip(*resource_result))[2] if x is not None
        ]
    except:
        pass

    cpu_resource_result = list(zip(*resource_result))

    try:
        resource_result = np.array(cpu_resource_result)[:2, :].astype(np.float32)
        cpu_ = int(10 * np.median(resource_result[0, :])) / 10
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
    
    print("------------------------------Summary------------------------------")
    print(f"tool's version:: {version}")
    print(f"num_clients:: {num_clients}")

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

def test_from_ids(forward_function: Union[
        Callable[[List[int]]], List[Callable[[List[int]]]]
    ],
    ids: List[int],
    request_batch=1):
    
    assert isinstance(forward_function, list)
    assert len(ids) > 0 
    assert isinstance(ids[0], int)
    total_number = len(ids)
    num_clients = len(forward_function)
    print(f'total_number={total_number}, num_clients={num_clients}')
    
    forwards = [LoopSampler(ids, request_batch) for i in range(num_clients)]
    for i in range(num_clients):
        forwards[i].forward = forward_function[i]

    return test(forwards, total_number)


def test_from_raw_file(
    forward_function: Union[
        Callable[[List[tuple[str, bytes]]]], List[Callable[[List[tuple[str, bytes]]]]]
    ],
    file_dir: str,
    num_clients=10,
    request_batch=1,
    total_number=10000,
    recursive=True,
    ext=[".jpg", ".JPG", ".jpeg", ".JPEG"],
):
    """
    This function is used to test the performance of a function.
    It can be used to test the performance of a function that processes a single image, or a function that processes a batch of images.
    """
    data = preload(
        file_dir=file_dir, recursive=recursive, ext=ext
    )

    print(f"file_dir = {file_dir}, num_clients = {num_clients}, request_batch = {request_batch}, total_number = {total_number}")
    assert len(data) > 0
    if total_number == 0:
        total_number = len(data)

    assert total_number > 0

    if isinstance(forward_function, list):
        assert len(forward_function) == num_clients
    else:
        forward_function = [forward_function] * num_clients

    forwards = [LoopSampler(data, request_batch) for i in range(num_clients)]
    for i in range(num_clients):
        forwards[i].forward = forward_function[i]

    return test(forwards, total_number)


def test_function(
    forward_function: Union[Callable, List[Callable]],
    num_clients=10,
    request_batch=1,
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
        FunctionSampler(forward_function[i], request_batch) for i in range(num_clients)
    ]
    return test(forwards, total_number)

def example_mutil_clients_speed_test(host, port, num_clients, file_dir, total_number):
    class Client:
        """wrapper for thrift's python API. You may need to re-implement this class."""
        def __init__(self, host, port, request_batch, id2data) -> None:
            """
            :param host: ip
            :type host: str
            :param port: port
            :type port: int
            :param request_batch: size of sended data in batches.
            :type request_batch: int
            """
            ## example thrift service:
            import sys
            sys.path.append("src")
            from api import InferenceService

            from api.ttypes import InferenceParams

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
            self.request_batch = request_batch
            self.id2data=id2data

        def forward(self, ids: List[int]) :
            """batch processing
            """
            assert len(ids) == 1, "right now only support bs = 1"
            
            ids[0] = self.id2data[ids[0]]
            
            result = self.client.infer_batch([self.InferenceParams(*x) for x in ids])
            

        def __del__(self):
            self.transport.close()
        
    # prepare_data
    ext = [".jpg", ".JPG", ".jpeg", ".JPEG"]
    list_images = [
            x for x in os.listdir(file_dir) if os.path.splitext(x)[-1] in ext
        ]
    list_images = [os.path.join(file_dir, x) for x in list_images]
    if len(list_images) > 1000:
        list_images = list_images[:1000] # not enough memory. todo cal mem online
    id2data = {}
    for i in range(len(list_images)):
        with open(list_images[i], 'rb') as f:
            id2data[i] = (list_images[i], f.read())
    print(f'{len(id2data)} file readed')
    ids = list(range(len(list_images)))
    repeats = math.ceil(total_number / len(ids))
    ids = (ids * repeats)[:total_number]
    
    instances = [Client(host, port, 1, id2data) for i in range(num_clients)]
    
    test_from_ids(
        forward_function=[x.forward for x in instances],
        ids=ids)


def example_mutil_clients_inference(host, port, num_clients, file_dir):
    final_result = {}
    class Client:
        """wrapper for thrift's python API. You may need to re-implement this class."""
        def __init__(self, host, port, request_batch, data) -> None:
            """
            :param host: ip
            :type host: str
            :param port: port
            :type port: int
            :param request_batch: size of sended data in batches.
            :type request_batch: int
            """
            ## example thrift service:
            import sys
            sys.path.append("src")
            from api import InferenceService

            from api.ttypes import InferenceParams

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
            self.request_batch = request_batch
            self.data = data

        def forward(self, data: List[int]) :
            """batch processing
            """
            assert len(data) == 1, "right now only support bs = 1"
            
            file_path = self.data[data[0]]
            with open(file_path, "rb") as f:
                data[0] = (file_path, f.read())
            result = self.client.infer_batch([self.InferenceParams(*x) for x in data])
            
            final_result[data[0][0]] = result[0]

        def __del__(self):
            self.transport.close()
        
    # prepare_data
    ext = [".jpg", ".JPG", ".jpeg", ".JPEG"]
    list_images = [
            x for x in os.listdir(file_dir) if os.path.splitext(x)[-1] in ext
        ]
    list_images = [os.path.join(file_dir, x) for x in list_images]
    
    instances = [Client(host, port, 1, list_images) for i in range(num_clients)]
    
    test_from_ids(
        forward_function=[x.forward for x in instances],
        ids=list(range(len(list_images)))
    )
    return final_result
  
