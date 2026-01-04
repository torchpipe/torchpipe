# throughput and lantecy test

# from curses import flash
import torch
# import torch_npu

from timeit import default_timer as timer
import cv2
import sys
import random
import os
import threading
import numpy as np

# from .device_tools import install_package

version = "20230104.0"

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
# update 0.1.7 2022-06-17 可设置pid；pid不存在时提示并退出；cpu利用率过小时，不显示结果（大概率匹配到错误的进程）fix request_batch>1   by zsy wlc
# update 0.1.8 2022-07-28  直接读取jpg binary， 不在预先解码（文件名后缀需要为 ".jpg", '.JPG', '.jpeg', '.JPEG'）
# update 2022-09-08  增加callback， 用于接收和处理结果；此时推理函数返回类型需要是list类型；
#                           当 total_number <=0 时，变为只跑一遍的模式；
#                           图片数量过少时，不再宕机;   
# update 2023-01-04  增加 PRELOAD_TYPE: yes no auto;                        

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
# 最新版本请访问 https://g.hz.netease.com/deploy/torchpipe/-/blob/master/torchpipe/tool/test_tools.py


max_pic_loaded = 999
infer_prob_print = True

PRELOAD_TYPE = os.environ.get("PRELOAD_TYPE", "auto")  # auto   no  yes


class TestParams:
    def __init__(self):
        self.random_choice = True


def pre_resize(im, max_size=640.0):
    h, w, _ = im.shape
    if max(h, w) <= max_size:
        return im
    ratio = max_size / max(h, w)

    resized_w = int(ratio * w)
    resized_h = int(ratio * h)
    im = cv2.resize(im, dsize=(resized_w, resized_h))
    return im


class InferThread(threading.Thread):
    def __init__(
        self,
        index,
        params,
        infer,
        tran_list,
        test_params: TestParams,
        error_time_out=5000,
    ) -> None:
        threading.Thread.__init__(self, name=str(index))
        self.params = params
        if self.params[0] == -1:
            self.params[0] = len(tran_list)
        assert self.params[0] >= len(tran_list)
        self.index = index

        self.infer = infer
        self.print_prob = self.params[4]

        self.tran_list = tran_list
        self.error_time_out = error_time_out
        self.callback = None
        self.test_params = test_params

    def set_callback(self, callback):
        self.callback = callback

    def run(self):
        while True:
            input = []
            with self.params[1]:
                self.params[0] = self.params[0] - 1 * self.params[3]
                if self.params[0] % 2000 == 0:
                    print(f"{self.params[0]}/... left", flush=True)
                    # sys.stdout.flush()
                # print(self.params[0])
                if self.params[0] < 0:
                    break

                for i in range(self.params[3]):
                    if len(self.tran_list):
                        if self.test_params.random_choice:
                            img_path, data = random.choice(self.tran_list)
                        else:
                            img_path, data = self.tran_list.pop()
                        # if data is None:
                        #     with open(img_path, 'rb') as f:
                        #         data=f.read()
                        input.append((img_path, data))

            for i in range(len(input)):
                if input[i][1] is None:
                    with open(input[i][0], "rb") as f:
                        data = f.read()
                    input[i] = (input[i][0], data)
            z = self.forward(input, infer_prob_print)
            for i in range(self.params[3]):
                self.params[2].append(z)

    def forward(self, input, out_print=True):
        if not input:
            return 0
        start = timer()
        result = self.infer(input)
        result_time = (timer() - start) * 1000
        if self.callback:
            for i in range(len(input)):
                img_path = input[i][0]
                tmp_reuslt = None
                if len(result) > i:
                    tmp_reuslt = result[i]
                else:
                    print(f"error: len(result) <= i. len(result)={len(result)}")
                self.callback(img_path, tmp_reuslt)

        if (
            out_print and random.random() < self.print_prob and result is not None
        ) or result_time > self.error_time_out:
            with self.params[1]:
                # sys.stdout.write(str(result))
                # sys.stdout.flush()
                if random.random() < self.params[5]:
                    print_str = str(result)
                    if len(print_str) > 10000:
                        self.params[5] /= 100
                    if len(print_str) > 1000:
                        self.params[5] /= 10
                    if len(print_str) > 100:
                        self.params[5] /= 2

                    if len(print_str) <= 60:
                        print_str += " " * (60 - len(print_str))
                    else:
                        print_str = print_str[: 60 - 5] + " ...."

                    print(print_str, end="\r")
        if result_time > self.error_time_out:
            print("response too slow: ", result_time, f"data left: {self.params[0]}")
        return result_time

    def __del__(self):
        pass

    def warmup(self, num):
        for i in range(num):
            self.forward(None, out_print=False)


class InferThreadData(threading.Thread):
    def __init__(
        self, index, params, infer, test_params: TestParams, error_time_out=5000
    ) -> None:
        threading.Thread.__init__(self, name=str(index))
        self.params = params

        self.index = index

        self.infer = infer
        self.print_prob = self.params[4]

        self.error_time_out = error_time_out
        self.callback = None
        self.test_params = test_params

    def set_callback(self, callback):
        self.callback = callback

    def run(self):
        while True:
            input = []
            with self.params[1]:
                self.params[0] = self.params[0] - 1 * self.params[3]
                if self.params[0] % 2000 == 0:
                    print(f"{self.params[0]}/... left", flush=True)
                    # sys.stdout.flush()
                # print(self.params[0])
                if self.params[0] < 0:
                    break

            z = self.forward(infer_prob_print)
            for i in range(self.params[3]):
                self.params[2].append(z)

    def forward(self, out_print=True):
        start = timer()
        result = self.infer()
        result_time = (timer() - start) * 1000
        if self.callback:
            self.callback(result)

        if (
            out_print and random.random() < self.print_prob and result is not None
        ) or result_time > self.error_time_out:
            with self.params[1]:
                # sys.stdout.write(str(result))
                # sys.stdout.flush()
                if random.random() < self.params[5]:
                    print_str = str(result)
                    if len(print_str) > 10000:
                        self.params[5] /= 100
                    if len(print_str) > 1000:
                        self.params[5] /= 10
                    if len(print_str) > 100:
                        self.params[5] /= 2

                    if len(print_str) <= 60:
                        print_str += " " * (60 - len(print_str))
                    else:
                        print_str = print_str[: 60 - 5] + " ...."

                    print(print_str, end="\r")
        if result_time > self.error_time_out:
            print("response too slow: ", result_time, f"data left: {self.params[0]}")
        return result_time

    def __del__(self):
        pass

    def warmup(self, num):
        for i in range(num):
            self.forward(False)


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
        while self.need_record_index < 0:
            i += 1
            i %= gpuDeviceCount
            handle = pynvml.nvmlDeviceGetHandleByIndex(
                i
            )  # 获取GPU i的handle，后续通过handle来处理
            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
            ## gpu_memory_total = info.total #GPU i的总显存
            # gpu_memory_used = info.used / NUM_EXPAND #转为MB单位
            # all_gpu_used.append(gpu_memory_used) #添加进list

            ###还可以直接针对pid的gpu消耗进行统计
            info_list = pynvml.nvmlDeviceGetComputeRunningProcesses(
                handle
            )  # 获取所有GPU上正在运行的进程信息
            info_list_len = len(info_list)
            gpu_memory_used = 0
            if info_list_len > 0:  # 0表示没有正在运行的进程
                for info_i in info_list:
                    print(info_i.pid, pid)
                    if info_i.pid == pid:  # 如果与需要记录的pid一致
                        # gpu_memory_used += info_i.usedGpuMemory / NUM_EXPAND #统计某pid使用的总显存
                        self.need_record_index = info_i
                        break
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
        self.pynvml.nvmlShutdown()


# note 如果待测试函数有返回值，比如cuda上的tensor，有一定概率会copy到cpu并打印出来（初始概率下约打印10次，后续如果打印对象太大，则相应递减概率，但通常对性能影响小于千分之三
class ResourceThread(threading.Thread):
    def __init__(self, pid, result_list, my_event):
        if pid == 0:
            pid = os.getpid()
            print(f"auto set --pid={pid}. Reset it if necessary")

        threading.Thread.__init__(self, name="ResourceThread:" + str(pid))
        try:
            import psutil
        except:
            install_package("psutil")
            import importlib

            psutil = importlib.util.find_spec("psutil", package=None)
            import psutil as psutil
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
        # self.gpu = GpuInfo(pid)
        # try:
        #     self.gpu = GpuInfo(pid)
        #     self.gpu.get_gpu_device()
        # except Exception as e:
        #     print(" pynvml :", e)

    def get_cpu_mem(self):
        return self.p.cpu_percent(), self.p.memory_percent()

    def run(self):
        import time

        while not self.my_event.isSet():
            time.sleep(2)
            cpu_percent = self.p.cpu_percent()
            mem_percent = self.p.memory_percent()
            if cpu_percent > 0 and mem_percent > 0:
                # if self.gpu:
                #     util = self.gpu.get_pid_info()
                #     self.result_list.append((cpu_percent, mem_percent, util))
                # else:
                self.result_list.append((cpu_percent, mem_percent))


def test(
    forward_function,
    request_client=10,
    request_batch=1,
    total_number=10000,
    pid=0,
    extra_post=[],
    call_back=[],
):
    """_summary_

    :param forward_function: _description_
    :type forward_function: _type_
    :param request_client: _description_, defaults to 10
    :type request_client: int, optional
    :param request_batch: _description_, defaults to 1
    :type request_batch: int, optional
    :param total_number: _description_, defaults to 1*1*10000
    :type total_number: _type_, optional
    :param pid: _description_, defaults to 0
    :type pid: int, optional
    :param extra_post: _description_, defaults to []
    :type extra_post: list, optional
    """

    test_params = TestParams()

    assert total_number > 0

    list_latency = []
    print_prob = 10.0 / total_number
    shared_print_prob = 1.0

    params = [
        total_number,
        threading.Lock(),
        list_latency,
        request_batch,
        print_prob,
        shared_print_prob,
    ]

    if isinstance(forward_function, list):
        instance_threads = [
            InferThreadData(i, params, forward_function[i], test_params)
            for i in range(request_client)
        ]
    else:
        instance_threads = [
            InferThreadData(i, params, forward_function, test_params)
            for i in range(request_client)
        ]
    assert isinstance(call_back, list) or call_back is None
    if isinstance(call_back, list) and call_back:
        if len(call_back) == 1:
            call_back = call_back * request_client
        assert len(call_back) == request_client
        for i in range(request_client):
            instance_threads[i].set_callback(call_back[i])

    print("Warm-up start")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=request_client) as t:
        for thread_ in instance_threads:
            t.submit(thread_.warmup, request_client)

    torch.npu.synchronize()
    print("Warm-up finished")
    print("", flush=True)

    resource_result = []
    resource_event = threading.Event()
    resource_thread = ResourceThread(pid, resource_result, resource_event)

    resource_thread.start()

    start = timer()
    for i in instance_threads:
        i.start()

    for i in instance_threads:
        i.join()
    resource_event.set()

    torch.npu.synchronize()
    total_time = timer() - start
    resource_thread.join()

    resource_result = np.array(resource_result)
    try:
        cpu_ = int(10 * np.median(resource_result[:, 0])) / 10
        if cpu_ < 0.8 * 100:
            cpu_ = "-"
    except:
        cpu_ = "-"

    try:
        gpu_ = int(10 * np.median(resource_result[:, 2])) / 10
        if gpu_ < 0.5 * 100:
            gpu_ = "-"
    except:
        gpu_ = "-"

    print("resource every 2s:")
    print(resource_result)
    print("")

    if len(list_latency) != total_number:
        print("len(list_latency): ", len(list_latency))
        print(f"total_number: {total_number}")
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
    avg = round(1000 * total_time / total_number * request_batch * request_client, 2)

    print("------------------------------Summary------------------------------")
    print(f"tool's version:: {version}")
    print(f"request client:: {request_client}")
    print(f"request batch::  {request_batch}")
    if test_params.random_choice:
        print(f"total number::   {total_number}")
    else:
        print(f"total number::   {total_number}(no data preloading)")
    print(f"throughput::     qps:  {qps},   [qps:=total_number/total_time]")
    print(
        f"                 avg:  {avg} ms   [avg:=1000/qps*(request_batch*request_client)]"
    )

    print(f"latency::        TP50: {tp50}   TP90: {tp90}   TP99:  {tp99} ")
    print(
        f"                 MEAN: {mean}   -50,-40,-20,-10,-1: {tp_5},{tp_4},{tp_3},{tp_2},{tp_1} ms"
    )
    print(f"cpu::            usage: {cpu_}%")
    print(
        "-------------------------------------------------------------------",
        flush=True,
    )


def test_from_raw_jpg(
    forward_function,
    img_dir: str,
    request_client=10,
    request_batch=1,
    total_number=10000,
    pid=0,
    extra_post=[],
    call_back=[],
):
    """multiple threads test tool.

    :param forward_function: core function whom want to be tested
    :type forward_function: function or list of functions whose input is batch data: List[(id, data)]
    :param img_dir: directory containing pictures
    :type img_dir: str
    :param request_client: number of clients, defaults to 10
    :type request_client: int, optional
    :param request_batch: number of jpgs per request, defaults to 1
    :type request_batch: int, optional
    :param total_number: total number of data needed to be processed. defaults to 10000

            * if total_number > 0:
                - preloading at most 999 or `total_number` pictures from img_dir randomly
                - warm-up
                - repeatedly choose preloaded data and run `forward_function`.
            * if total_number==-1:
                - all data in img_dir will be tested.

    :type total_number: int, optional
    :param pid: process id needed to be monitored, defaults to 0
    :type pid: int, optional
    :param extra_post: extra supported suffix, for example `.png`, defaults to []
    :type extra_post: list, optional
    :param call_back: call-back functions whose input are (id, single_result), the length could be 1 or request_client, defaults to []
    :type call_back: list, optional
    """
    tran_list = []
    img_root = img_dir
    if not os.path.exists(img_root):
        print(img_dir, " not exists")
    # get images list
    print(
        f"\nstart testing, request_client={request_client}, request_batch={request_batch}, total_number={total_number}",
        flush=True,
    )
    local_max_pic_loaded = -1

    global PRELOAD_TYPE
    if PRELOAD_TYPE == "auto":
        if total_number > 0:
            PRELOAD_TYPE = "yes"
        else:
            PRELOAD_TYPE = "no"

    if total_number < 0:
        assert PRELOAD_TYPE == "no"

    if PRELOAD_TYPE != "no":
        assert total_number > 0
        local_max_pic_loaded = min(max_pic_loaded, total_number)
        print(
            f"preloading first {local_max_pic_loaded} images from {img_root} and its subdirectories",
            flush=True,
        )

    list_images = []
    for root, folders, filenames in os.walk(img_root):
        for filename in filenames:
            list_images.append(os.path.join(root, filename))
    all_post = [".jpg", ".JPG", ".jpeg", ".JPEG"] + extra_post
    for img_path in list_images:
        if not os.path.splitext(img_path)[-1] in all_post:
            print(f"skip {img_path}")
            continue

        # img = cv2.imread(img_path, 1)
        # if img is None:
        #     continue
        # img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[
        #     1].tobytes()
        # img=pre_resize(img)
        if local_max_pic_loaded > 0 and len(tran_list) > local_max_pic_loaded:
            continue

        img = None

        if PRELOAD_TYPE == "yes":
            with open(img_path, "rb") as f:
                img = f.read()
        tran_list.append((img_path, img))

    test_params = TestParams()

    if total_number <= 0:
        assert PRELOAD_TYPE == "auto" or PRELOAD_TYPE == "no"
        total_number = len(tran_list)
        test_params.random_choice = False

    list_latency = []
    print_prob = 10.0 / total_number
    shared_print_prob = 1.0

    params = [
        total_number,
        threading.Lock(),
        list_latency,
        request_batch,
        print_prob,
        shared_print_prob,
    ]

    if isinstance(forward_function, list):
        instance_threads = [
            InferThread(i, params, forward_function[i], tran_list, test_params)
            for i in range(request_client)
        ]
    else:
        instance_threads = [
            InferThread(i, params, forward_function, tran_list, test_params)
            for i in range(request_client)
        ]
    assert isinstance(call_back, list) or call_back is None
    if isinstance(call_back, list) and call_back:
        if len(call_back) == 1:
            call_back = call_back * request_client
        assert len(call_back) == request_client
        for i in range(request_client):
            instance_threads[i].set_callback(call_back[i])

    print("Warm-up start")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=request_client) as t:
        for thread_ in instance_threads:
            t.submit(thread_.warmup, request_client)

    # torch.npu.synchronize()
    print("Warm-up finished")
    print("", flush=True)

    resource_result = []
    resource_event = threading.Event()
    resource_thread = ResourceThread(pid, resource_result, resource_event)

    resource_thread.start()

    start = timer()
    for i in instance_threads:
        i.start()

    for i in instance_threads:
        i.join()
    resource_event.set()

    # torch.npu.synchronize()
    total_time = timer() - start
    resource_thread.join()

    resource_result = np.array(resource_result)
    try:
        cpu_ = int(10 * np.median(resource_result[:, 0])) / 10
        if cpu_ < 0.8 * 100:
            cpu_ = "-"
    except:
        cpu_ = "-"

    try:
        gpu_ = int(10 * np.median(resource_result[:, 2])) / 10
        if gpu_ < 0.5 * 100:
            gpu_ = "-"
    except:
        gpu_ = "-"

    print("resource every 2s:")
    print(resource_result)
    print("")

    if len(list_latency) != total_number:
        print("len(list_latency): ", len(list_latency))
        print(f"total_number: {total_number}")
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
    avg = round(1000 * total_time / total_number * request_batch * request_client, 2)

    print("------------------------------Summary------------------------------")
    print(f"tool's version:: {version}")
    print(f"request client:: {request_client}")
    print(f"request batch::  {request_batch}")
    if test_params.random_choice:
        print(f"total number::   {total_number}")
    else:
        print(f"total number::   {total_number}(no data preloading)")
    print(f"throughput::     qps:  {qps},   [qps:=total_number/total_time]")
    print(
        f"                 avg:  {avg} ms   [avg:=1000/qps*(request_batch*request_client)]"
    )

    print(f"latency::        TP50: {tp50}   TP90: {tp90}   TP99:  {tp99} ")
    print(
        f"                 MEAN: {mean}   -50,-40,-20,-10,-1: {tp_5},{tp_4},{tp_3},{tp_2},{tp_1} ms"
    )
    print(f"cpu::            usage: {cpu_}%")
    print(
        "-------------------------------------------------------------------",
        flush=True,
    )


class ThriftInfer:
    """wrapper for thrift's python API. You may need to re-implement this class."""

    def __init__(self, host, port, request_batch) -> None:
        """

        :param host: ip
        :type host: str
        :param port: port
        :type port: int
        :param request_batch: size of sended data in batches.
        :type request_batch: int
        """

        import sys
        sys.path.append("src")
        from api import InferenceService

        from api.ttypes import InferenceResult, InferenceStatusEnum, InferenceResultUnit, InferenceParams


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


def test_thrift_from_raw_jpg(
    img_dir,
    host="localhost",
    port=8095,
    request_client=10,
    request_batch=1,
    total_number=10000,
):
    instances_ = [ThriftInfer(host, port, request_batch) for i in range(request_client)]

    test_from_raw_jpg(
        [x.infer for x in instances_],
        img_dir,
        request_client,
        request_batch,
        total_number,
    )


def parse_arguments(argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="Port to listen.", default=9002)
    parser.add_argument(
        "--host", type=str, help="Host to run service", default="localhost"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        help=f"img path. 预读取该目录以及子目录下至多{max_pic_loaded}张图片",
        default="img/",
    )
    parser.add_argument("--request_batch", type=int, help="单次请求的数据量", default=1)
    parser.add_argument("--request_client", type=int, help="并发请求数", default=10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    total_number = 10000

    args = parse_arguments(sys.argv[1:])

    test_thrift_from_raw_jpg(
        args.img_dir,
        host=args.host,
        port=args.port,
        request_client=args.request_client,
        request_batch=args.request_batch,
        total_number=total_number,
    )
