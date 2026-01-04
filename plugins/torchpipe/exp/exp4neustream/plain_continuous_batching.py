
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

config_parser = exp_config_parser.ConfigParser()
# latency_profile = exp_config_parser.get_latency_profile()

# omniback.init("DebugLogger")


torch.set_grad_enabled(False)

sys.path.insert(0, './')
assert os.environ["USE_TRT"] == "True"


from trt_modules import ClipModule, UNetModule, VaeModule, SafetyModule

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

        for key, value in config.items():
            # if key == 'UNetModule':
            #     continue
            self.modules[key] = self.modules_type[key](**value)
            # self.modules[key].deploy()
        self.default_deploy()
    def default_deploy(self, **kwargs):
        # for module in self.stream_module_list:
        for k, v in self.modules.items():
            v.deploy()


# 初始化管道
sd_config_file = "stable_diffusion_v1_5/config.json"
sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)

sd_modules = sd_pipeline.modules


class Clip:
    def init(self, params, options):
        torch.set_grad_enabled(False)

    def forward(self, io: List[omniback.Dict]):
        reqs = []
        for data in io:
            id = data['data']
            req = global_request_pool[id]
            data['result'] = id
            reqs.append(req)
        # print(f'{id}: clip bs = {len(io)}, time={time.time()}')
        sd_modules['ClipModule'].compute(reqs)
        # torch.cuda.current_stream().synchronize()
        # print(f'clip done: {id}, time={time.time()}')

    def max(self):
        # print('xxxxxxxxxxxxxxd')
        return 16


class Unet:
    def init(self, params, options):
        print(f'unet init: {torch.cuda.current_stream()}')
        torch.set_grad_enabled(False)
        print(f'unet params={params}')
        self.sd_module = sd_modules['UNetModule']

    def forward(self, io: List[omniback.Dict]):
        # print(f'unet forard: {torch.cuda.current_stream()}')
        
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
            # index = req["loop_index"]["UNetModule"]
            # print(f"{id}/{index}: restart to {data['restart']}")
            data['result'] = id
        # print(f'{id}: unet bs = {len(io)}, time={time.time()}')
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
        return 16
    
omniback.register("Clip", Clip)
omniback.register("Unet", Unet)
omniback.register("Vae", Vae)

@dataclass
class LoopInfo:
    data: omniback.Dict = None
    event: omniback.Event = None
    loop_index: int = 0
    loop_num: int = 1
    emergency: float = 0.0
    emergency_threshold: float = 10000
    iter_deadline: float = 0.0
    time: float = 0.0

class PyContinuousBatching:
    def init(self, params, options):
        print(f'(ContinuousBatching)init foom {params}')
        self.receiving_data = {}
        self._stop_event = threading.Event()
        
        self.cached_data = {}
        self.target = omniback.get(params.pop('target'))

        self.max_batch_size = int(params.pop('max_batch_size'))

        self.running_ids = set()
        self.input_queue = queue.Queue()
        
        self.thread = threading.Thread(target=self.task_loop, daemon=True)
        self.thread.start()

    def all_received(self):
        return all(local_id in self.receiving_data for local_id in self.running_ids)

    def __del__(self):
        self._stop_event.set()
        self.thread.join(timeout=5.0) 
        
    def forward(self, io: List[omniback.Dict]):
        self.input_queue.put(('start', io))

    def task_loop(self):
        while not self._stop_event.is_set():
            # all in one thread !!!! no race condition
            try:
                step, task = self.input_queue.get(timeout=0.1)
                if step == 'start':
                    self.on_start(task)
                elif step == "finish":
                    self.on_finish(task)
                    if self.input_queue.qsize() == 0 and len(self.receiving_data) != 0: # 检查残余数据
                        self.on_start([]) # recheck if all received
                else:
                    raise RuntimeError(f'error step {step}')

                self.input_queue.task_done()

            except queue.Empty:
                continue

    def on_start(self, io):
        for data in io:
            self.receiving_data[data['data']] = data
        
        if not self.all_received():
            return
        else:
            receiving_data = self.receiving_data
            self.receiving_data = {}

        for req_id, data in receiving_data.items():
            loop_num = data['loop_num']

            if not req_id in self.cached_data:
                ev = data.pop('event')
                if data.contains('iter_deadline'):
                    self.cached_data[req_id] = LoopInfo(
                        data, ev, 0, loop_num=loop_num,
                        time=time.time(), 
                        iter_deadline=data['iter_deadline'],
                        emergency_threshold=data['emergency_threshold'])
                else:
                    self.cached_data[req_id] = LoopInfo(
                        data, ev, 0, loop_num=loop_num,
                        time=time.time())
            else:
                ev = data.pop('event')
                self.cached_data[req_id].data = data
                self.cached_data[req_id].event = ev
                self.cached_data[req_id].loop_index += 1
                self.cached_data[req_id].loop_num = loop_num

        receiving_data.clear()

        need_drop = set()
        for req_id, data in self.cached_data.items():
            left_round = data.loop_num - data.loop_index
            iter_deadline = data.iter_deadline
            if iter_deadline != 0:
                left_time = max(1e-10, iter_deadline - time.time())
                data.emergency = left_round / left_time  # 40 轮 3.5 秒
                if data.emergency > data.emergency_threshold:
                    need_drop.add(req_id)
                    print(f'iter_deadline={iter_deadline},now={time.time()}')
                    print(
                        f'drop(no enough time) {req_id} emergency = {data.emergency}/{data.emergency_threshold}, loop_index = {data.loop_index}/{data.loop_num}, left_time = {left_time}')
            
        self.running_ids = set()
        self.drop_request(need_drop)
        delayed = self.delay_some_request()
        self.drop_request(delayed)
        
        if len(self.cached_data) == 0: # impossible
            print('warning: empty data (all dropped) in continuous batching')
            return

        self.running_ids = set(x for x in self.cached_data.keys())
        if len(delayed) > 0:
            print(f'run: {self.running_ids}; removed = {delayed}')

        io = []
        for k in self.running_ids:
            v = self.cached_data[k]
            io.append(v.data)
            assert 'event' not in v.data
            
        ev = omniback.Event(len(io))
        ev.set_callback(lambda: self.input_queue.put(('finish', None)))
        ev.set_exception_callback(lambda x: print(f'error x')) # not allowed in continuous batching
        for item in io:
            item['event'] = ev
        self.target.forward(io, None)  

    def on_finish(self, io):
        should_del = []
        for k, v in self.cached_data.items():
            if k in self.running_ids:
                if v.loop_index + 1 >= v.loop_num:
                    should_del.append(k)

                v.data['event'] = v.event
                v.event.notify_all()

        for req_id in should_del:
            del self.cached_data[req_id]
            self.running_ids.discard(req_id)
            print(f'request {req_id} finished in iteration')

    def delay_some_request(self):
        delayed = set()
        need_delay = len(self.cached_data) - self.max_batch_size
        if need_delay <= 0:
            return delayed

        for req_id, v in heapq.nlargest(need_delay, self.cached_data.items(), key=lambda x: x[1].time):
            delayed.add(req_id)

        return delayed

    def drop_request(self, req_ids):
        for req_id in req_ids:
            v = self.cached_data[req_id]
            print(f'drop single {req_id}')

            v.data['event'] = v.event
            v.event.notify_all()
            
            del self.cached_data[req_id]
            del global_request_pool[req_id]

omniback.register("PyContinuousBatching", PyContinuousBatching)


def get_scheduler(max_batch_size):
    omniback.pipe({'unet_backend': {'backend': 'S[Unet,SyncTensor]', 'batching_timeout': 1, 'instance_num': 1},
               })

    config = {
        'global': {'entrypoint': 'Restart[DagDispatcher]'},
        'clip': {'backend': 'S[Clip,SyncTensor]', 'batching_timeout': 4, 'instance_num': 1, 'next': 'unet'},
        'unet': {'node_entrypoint': 'Register[PyContinuousBatching]', 'target': 'node.unet_backend', 'max_batch_size': max_batch_size},
        'vaesafety': {"backend": 'S[Vae,SyncTensor]', 'batching_timeout': 4, 'instance_num': 1}}

    scheduler = omniback.pipe(config)

    return scheduler


class Model:
    def __init__(self, slo_factor=19):
        # omniback.init('DebugLogger')
        max_batch_size = config_parser.get_max_batch_size(int(slo_factor*10))
        self.scheduler = get_scheduler(max_batch_size)

    def __call__(self, req_id, request, exc_cb, call_back):
        event = omniback.Event()
        event.set_exception_callback(exc_cb)
        
        def drop_cb():
            if request['loop_index']['UNetModule'] != request['loop_num']['UNetModule']:
                for k, v in request.items():
                    if k not in ['request_time', 'loop_index', 'loop_num', 'exception', 'id', "SLO"]:
                        request[k] = None

        event.set_callback(call_back)
        event.set_callback(drop_cb)
        
        emergency_threshold = request.pop('emergency_threshold', None)
        iter_deadline = request.pop('iter_deadline', None)
        io = {'data': req_id, 'node_name': 'clip', 'event': event,
              'loop_num': request['loop_num']["UNetModule"]}
        if iter_deadline is not None:
            io['iter_deadline'] = iter_deadline
            assert emergency_threshold is not None
            io['emergency_threshold'] = emergency_threshold
        io = omniback.Dict(io)

        assert req_id not in global_request_pool
        global_request_pool[req_id] = request

        self.scheduler(io)


