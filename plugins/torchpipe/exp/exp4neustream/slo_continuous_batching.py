
from modules import ClipModule, UNetModule, VaeModule, SafetyModule
import modules
import sys
from helper import exp_config_parser
import toml
import json
import torch
import omniback
import torchpipe
from typing import List
import time
import random
import os
import math
from dataclasses import dataclass
import threading
import heapq
import numpy as np

import queue



global_request_pool = {}
# NORM_TIME = 1
torch.set_grad_enabled(False)

# omniback.init("DebugLogger")

latency_profile = exp_config_parser.get_latency_profile()

torch.set_grad_enabled(False)

sys.path.insert(0, './')
assert os.environ["USE_TRT"] == "True"

class StableDiffusionPipeline():
    def __init__(self, config_path, **kwargs):
        super().__init__()
        fp = open(config_path, "r")
        config = json.load(fp)
        self.stream_module_list = []
        print(config.keys())
        torch.set_grad_enabled(False)

        self.modules_type = {'ClipModule': ClipModule,
                             'UNetModule': UNetModule,
                             'VaeModule': VaeModule,
                             'SafetyModule': SafetyModule}
        self.modules = {}
        for instance_index in range(1):
            for key, value in config.items():
                if key != 'UNetModule':
                    continue
                value['instance_index'] = instance_index
                nk = key + f"{instance_index}"
                self.modules[nk] = self.modules_type[key](**value)
        for instance_index in range(1):
            for key, value in config.items():
                if key != 'UNetModule':
                    continue
                value['instance_index'] = instance_index
                nk = key + f"{instance_index}"
                self.modules[nk].deploy(**value)

        for key, value in config.items():
            if key == 'UNetModule':
                continue
            self.modules[key] = self.modules_type[key](**value)
            self.modules[key].deploy()

    def default_deploy(self, **kwargs):
        for module in self.stream_module_list:
            module.deploy()


# 初始化管道
sd_config_file = "stable_diffusion_v1_5/config.json"
sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)

sd_modules = sd_pipeline.modules


class Clip:
    def init(self, params, options):
        torch.set_grad_enabled(False)

    def forward(self, io: List[omniback.Dict]):
        # print(f'clip bs ={len(io)}')
        reqs = []
        for data in io:
            id = data['data']

            req = global_request_pool[id]
            # print('cl112p', type(id))
            data['result'] = id
            # data['stage'] = 1
            # print('cli3p')
            reqs.append(req)

        sd_modules['ClipModule'].compute(reqs)

    def max(self):
        # print('xxxxxxxxxxxxxxd')
        return 16


class Unet:
    def init(self, params, options):
        torch.set_grad_enabled(False)
        print(f'unet params={params}')
        self.sd_module = sd_modules['UNetModule']

    def forward(self, io: List[omniback.Dict]):
        # print(f'unet bs = {len(io)}')
        # start_time = time.perf_counter()
        reqs = []
        for data in io:
            id = data['data']
            req = global_request_pool[id]
            for key in req.keys():
                if type(req[key]) == torch.Tensor and req[key].is_cpu:
                    req[key] = req[key].to('cuda')

            reqs.append(req)

            if req["loop_index"]["UNetModule"] + 1 < req['loop_num']["UNetModule"]:
                data['restart'] = 'unet'
            elif req["loop_index"]["UNetModule"] + 1 == req['loop_num']["UNetModule"]:
                data['restart'] = 'vaesafety'
            data['result'] = id

            # print(req)
        self.sd_module.compute(reqs)

        for req in reqs:
            req["loop_index"]["UNetModule"] += 1
        # print(io)
        # print(f'unet time = {time.perf_counter()-start_time}')

    def max(self):
        print('xxxxxxxxxxxxxxd4')
        return 10000

class Vae:
    def init(self, params, options):
        torch.set_grad_enabled(False)

    def forward(self, io: List[omniback.Dict]):
        # print('vae1')
        # print(f'vae bs ={len(io)}')
        # start_time = time.perf_counter()
        reqs = []
        for data in io:
            req_id = data['data']
            req = global_request_pool[req_id]
            data['result'] = req_id
            reqs.append(req)
            del global_request_pool[req_id]
        sd_modules['VaeModule'].compute(reqs)
        sd_modules['SafetyModule'].compute(reqs)

        for req in reqs:
            for k, v in req.items():
                if type(v) == torch.Tensor:
                    req[k] = None  # v.cpu()

    def max(self):
        print('xxxxxxxxxxxxxxd')
        return 32


@dataclass
class LoopInfo:
    data: omniback.Dict = None
    event: omniback.Event = None
    loop_index: int = 0
    loop_num: int = 1
    emergency: float = 0.0
    emergency_v2: float = 0.0
    delay: int = 0
    time: float = 0
    max_batch_size: int = 0
    # start_iteration_time: float = 0
    # stop_iteration_time: float = 0


class PyContinuousBatching:
    def init(self, params, options):
        print(f'init foom {params}')
        self.receiving_data = {}
        self.lock = threading.Lock()

        self.cached_data = {}
        self.target = omniback.get(params.pop('target'))

        slo_factor = params.pop('slo_factor')
        print(f'params.pop(slo_factor) = {slo_factor}')
        slo_factor = [float(x) for x in slo_factor.split(',')]
        if len(slo_factor) == 2:
            slo_factor = slo_factor[0]
        else:
            slo_factor = slo_factor[0]
        self.slo_factor = float(slo_factor)

        self.config_parser = exp_config_parser.ConfigParser(step_delta=0.95)

        self.max_batch_size = self.config_parser.get_max_batch_size(
            int(10*self.slo_factor))
        # self.scheduler = self.config_parser.get_scheduler()
        self.instance_num = 1

        # self.waiting_ids = set()
        print(
            f'PyContinuousBatching max_batch_size = {self.max_batch_size} {self.config_parser.get_slo_factor_to_batch_size()} slo_factor= {self.slo_factor}')
        self.delayed = set()
        # self.start_batching = True

        self.input_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self.task_loop, daemon=True)
        self.thread.start()

    def all_received(self):
        for item in self.cached_data.keys():
            if item not in self.receiving_data and item not in self.delayed:
                # print(f'item={item} not received')
                return False
        return True

    def __del__(self):
        self._stop_event.set()
        self.thread.join(timeout=5.0)  # 超时保护
        if self.thread.is_alive():
            print("Warning: Worker thread did not exit cleanly")

    def forward(self, io: List[omniback.Dict]):
        self.input_queue.put(('forward', io))

    def task_loop(self):
        while not self._stop_event.is_set():
            # all in one thread !!!! no race condition
            try:
                step, task = self.input_queue.get(timeout=0.1)
                if step == 'forward':
                    self.on_start(task)
                elif step == "finish":
                    self.on_finish(task)
                else:
                    print(f'error step {step}')

                self.input_queue.task_done()

            except queue.Empty:
                continue

    def on_start(self, io):
        for data in io:
            # print(data['data'])
            # data['data'] is the request ID, and it's user's 责任去让他与实际数据一一对应
            self.receiving_data[data['data']] = data

        if not self.all_received():
            return
        else:
            receiving_data = self.receiving_data
            self.receiving_data = {}

        # all data received. update receiving_data to cached_data (receiving_data >= cached).
        # first_round_data = []
        for req_id, data in receiving_data.items():
            # print(f'parsing {req_id}')
            req = global_request_pool[req_id]  # only for debug
            # import pdb;pdb.set_trace()
            loop_num = data['loop_num']
            if not req_id in self.cached_data:
                # assert req['loop_index']['UNetModule'] == 0
                # assert req['loop_num']['UNetModule'] == loop_num
                req['start_iteration_time'] = time.time()
                ev = data.pop('event')
                self.cached_data[req_id] = LoopInfo(
                    data, ev, 0, loop_num=loop_num, time=req['start_iteration_time'])
                # first_round_data.append(req_id)
            else:
                # assert 'event' in data
                ev = data.pop('event')
                self.cached_data[req_id].data = data
                self.cached_data[req_id].event = ev
                self.cached_data[req_id].loop_index += 1
                # can be changed per iteration
                self.cached_data[req_id].loop_num = loop_num
                # assert self.cached_data[req_id].loop_index == req['loop_index']['UNetModule']
                # print(f'{req_id}: {self.cached_data[req_id].loop_index}/{self.cached_data[req_id].loop_num}')
        # clear data
        # print(receiving_data, self.cached_data)
        receiving_data.clear()

        need_drop = set()
        all_batch_sizes = []
        for req_id, data in self.cached_data.items():
            req = global_request_pool[req_id]
            # compute emergency
            SLO = req['SLO']
            start_iteration_time = req['start_iteration_time']
            request_time = req['request_time']
            # profile_post = self.profile_post[min(len(self.cached_data), self.max_batch_size)]
            # profile_post = self.profile_post[1]
            # profile_post = 0.010051007685251533 + 0.0417292199190706  # from profile

            profile_post = latency_profile['vae'][1] + \
                latency_profile['safety'][1]  # + 0.06
            SLO_iter = SLO - (start_iteration_time -
                              request_time) - profile_post
            left_time = max(1e-6, SLO_iter -
                            (time.time() - start_iteration_time))
            left_round = data.loop_num - data.loop_index
            slo_factor = (left_time + profile_post) / \
                (latency_profile["unet"][1]*left_round + profile_post)
            # slo_factor = left_time/(latency_profile["unet"]['1']*left_round)

            print(
                f'left_time={left_time}, left_rd={data.loop_index}/{data.loop_num} slo_factor={slo_factor}')
            new_slo_factor = int(math.floor(slo_factor * 10))
            # emv3 = left_round * float(latency_profile['unet']['1'])/ left_time # from profile

            if new_slo_factor < 12:  # emv3 >= 1:# or data.emergency_v2 > 1.8:
                # 14.12/1.44 16.28/1.56
                need_drop.add(req_id)
                bs = 0
                data.max_batch_size = 0
            elif new_slo_factor >= 11:
                bs = self.config_parser.get_max_batch_size(new_slo_factor)
                # import pdb;pdb.set_trace()
                # bs = self.slo_factor_to_batch_size[new_slo_factor]
                # all_batch_sizes.append((req_id, bs))
                # if bs <= 15 and self.cached_data[req_id].delay == 0:
                #     print(f'warning: bs={bs}, id={req_id} all={all_batch_sizes}, left_time={left_time} left_round={left_round}')
                data.max_batch_size = bs
            else:
                bs = 0
                data.max_batch_size = 0

            # if len(self.cached_data) - bs  > 0:
            #     print(f'ID={req_id} slo_factor={slo_factor:.2} MBS={bs} curr={len(self.cached_data)}')
            # print(f'slo_factor={slo_factor},bs={bs} index={data.loop_index}/{data.loop_num}')

            data.emergency = left_round / left_time  # 40 轮 3.5 秒

        self.drop_request(need_drop)
        if len(self.cached_data) == 0:
            print('warning: empty (dropped) data in continuous batching')
            return

        # self.max_batch_size = 20# = self.update_max_batch_size(all_batch_sizes_high, all_batch_sizes_low) + 0

        # print(f'{all_batch_sizes} self.max_batch_size={self.max_batch_size}')

        # judge should we put new request into the running loop, or just drop them
        if False:
            self.delayed = self.delay_some_request()
        else:
            # self.delay_some_request_by_slo_factor()
            self.delayed = self.delay_some_request()
            # self.delayed =

        # print(f'cb: start. {len(self.cached_data)}')
        # now run
        io = []
        for k, v in self.cached_data.items():
            if k not in self.delayed:
                io.append(v.data)
                assert 'event' not in v.data
                req = global_request_pool[k]
                # req['freeness'] = self.max_batch_size - len(self.cached_data)
                # print(req['freeness'])
                # print(f'running = {len(self.cached_data)}')

        ev = omniback.Event(len(io))
        ev.set_callback(lambda: self.input_queue.put(('finish', None)))
        ev.set_exception_callback(lambda x: print(f'error x'))
        for item in io:
            item['event'] = ev
        # print(f'cb forward {len(io)}')
        self.target.forward(io, None)  # can be async, but we choose sync here

    def on_finish(self, io):
        should_del = []
        for k, v in self.cached_data.items():
            if k not in self.delayed:
                if v.loop_index + 1 >= v.loop_num:
                    should_del.append(k)
                    req = global_request_pool[k]
                    # req['stop_iteration_time'] = stop_iteration_time
                    # req['stop_iteration_batch'] = len(self.cached_data)
                v.data['event'] = v.event
                v.event.notify_all()
                # print(f'request {k} notified')
        for req_id in should_del:
            del self.cached_data[req_id]
            print(f'request {req_id} finished')
        # with self.lock:
        #     self.start_batching = True

    # def ff(self, x):
    #     # e = self.cached_data[x].emergency
    #     # thres = 30
    #     left = self.cached_data[x].loop_num - self.cached_data[x].loop_index
    #     return left# * (1- thres / e)

    def ff(self, x):
        # return (x[1].delay, x[1].emergency)
        return (x[1].delay, x[1].time)
        # left / (time - x / thr) <= thr
        # =>
        # e = x[1].emergency
        # thres = 1/(self.time_one_loop +1e-8)
        # print(thres)
        left = x[1].loop_num - x[1].loop_index
        return left  # * (thres/min(e,thres) - 1)

    def delay_some_request(self):
        # old_delayed = self.delayed
        # thres = 8
        delayed = set()
        need_delay = len(self.cached_data) - self.max_batch_size
        if need_delay <= 0:
            return delayed
        # pool = set(x for x  in self.cached_data.keys() if x not in  old_delayed)
        # if need_delay <= len(old_delayed):
        #     pass
        # else:

        #     for req_id in heapq.nsmallest(need_delay-len(old_delayed), pool, key=lambda x: self.cached_data[x].emergency):

        # need_drop = 0
        # for req_id, v in heapq.nsmallest(need_delay, self.cached_data.items(), key=lambda x: x[1].emergency):
        #     if v.emergency > thres:
        #         need_drop += 1

        for req_id, v in heapq.nlargest(need_delay, self.cached_data.items(), key=self.ff):
            v.delay += 1
            delayed.add(req_id)
            # print(f'delay: id={req_id} times={v.delay}, MBS={self.max_batch_size} em={v.emergency}')

        return delayed

    def drop_request(self, req_ids):
        for req_id in req_ids:
            v = self.cached_data[req_id]
            print(
                f'drop single {req_id}. emergency = {v.emergency}/{v.emergency_v2}')

            v.data['event'] = v.event
            v.event.notify_all()

            del self.cached_data[req_id]
            del global_request_pool[req_id]

        return

omniback.register("Clip", Clip)
omniback.register("Unet", Unet)
omniback.register("Vae", Vae)
omniback.register("PyContinuousBatching", PyContinuousBatching)


def get_scheduler(slo_factor):
    # omniback.pipe({'unet_backend':{'backend':'S[unet, SyncTensor,TimeStamp(finish_iter)]','batching_timeout':1,'instance_num':1},
    #     })
    omniback.pipe({'unet_backend': {'backend': 'SyncTensor[Unet]', 'batching_timeout': 1, 'instance_num': 1},
               })

    config = {
        'global': {'entrypoint': 'Restart[DagDispatcher]'},
        'clip': {'backend': 'SyncTensor[Clip]', 'batching_timeout': 8, 'instance_num': 1, 'next': 'unet'},
        'unet': {'node_entrypoint': 'Register[PyContinuousBatching]', 'target': 'node.unet_backend', 'slo_factor': slo_factor},
        'vaesafety': {"backend": 'SyncTensor[Vae]', 'batching_timeout': 8, 'instance_num': 1}}

    scheduler = omniback.pipe(config)

    return scheduler


class Model:
    def __init__(self, slo_factor):
        # omniback.init('DebugLogger')
        if isinstance(slo_factor, tuple):
            assert len(slo_factor) == 2
            slo_factor = str(slo_factor[0]) + ','+str(slo_factor[1])
        self.scheduler = get_scheduler(slo_factor)

    def __call__(self, req_id, request, exc_cb, call_back):
        event = omniback.Event()
        # print('before set_exception_callback')
        event.set_exception_callback(exc_cb)
        # print('after set_exception_callback')

        def drop_cb():
            if request['loop_index']['UNetModule'] != request['loop_num']['UNetModule']:
                # print(f'drop_cb, id={request["id"]} {id(request)}')
                for k, v in request.items():
                    if k not in ['request_time', 'loop_index', 'loop_num', 'exception', 'id', "SLO"]:
                        request[k] = None
                # print(request)

        event.set_callback(call_back)
        event.set_callback(drop_cb)
        # print(f'id={req_id}')

        io = omniback.Dict({'data': req_id, 'node_name': 'continuous', 'event': event,
                       'stage': 0, 'loop_num': request['loop_num']["UNetModule"]})
        # io.pop('event', None)
        # print(io)
        assert req_id not in global_request_pool
        global_request_pool[req_id] = request

        # start = time.perf_counter()
        self.scheduler(io)
        # event.wait()
        # print(f'total = {time.perf_counter() - start}')


def main(save_img=False, num_request=100, image_size=256):
    prompts = [
        "a photograph of an astronaut riding a horse",
        "A cute otter in a rainbow whirlpool holding shells, watercolor",
        "An avocado armchair",
        "A white dog wearing sunglasses"
    ]
    prompt = random.choice(prompts)
    id = 0

    scheduler = get_scheduler()

    def run():
        request = {
            "prompt": random.choice(prompts),
            "height": image_size,
            "width": image_size,
            "loop_num": {"UNetModule": random.randint(30, 50)},
            "request_time": time.time(),
            "guidance_scale": 7.5,
            "seed": 42 + id,  # 每轮使用不同的种子
            "SLO": 10000,
            "loop_index": {"UNetModule": 0},
            "id": id
        }
        global_request_pool[id] = request

        io = {'data': id, 'node_name': 'continuous',
              'loop_num': request['loop_num']["UNetModule"]}
        scheduler(io)
        print(
            f"(id={request['id']}) Server-side end2end latency: {time.perf_counter() - request['request_time']}")

        return request

    for index in range(20):
        run()  # warm up
        print(f'warm up {index}')
    print('warm up finished')

    start = time.perf_counter()

    for _ in range(num_request):
        run()
    total_time = (time.perf_counter() - start)/num_request
    print(f"time: {total_time:.4f} second")

    # print(global_request_pool[id])
    if save_img:
        request = run()
        print(request.keys())
        img = request['pillow_image']
        img.save(f"output_image_run{id}.jpg", quality=95)
        print(f"saved to output_image_run{id}.jpg")


if __name__ == "__main__":
    main(save_img=True, num_request=100, image_size=256)
