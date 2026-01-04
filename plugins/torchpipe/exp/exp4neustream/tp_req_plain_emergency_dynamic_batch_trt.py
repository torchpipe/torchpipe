
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


class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            return self._dict.get(key)

    def set(self, key, value):
        with self._lock:
            self._dict[key] = value

    def delete(self, key):
        with self._lock:
            if key in self._dict:
                del self._dict[key]


global_request_pool = {}
# NORM_TIME = 1
torch.set_grad_enabled(False)

# omniback.init("DebugLogger")
# with open('data/test_config.toml', 'r') as f:
#     data = toml.load(f)
# EXP_ID = os.getenv('EXP_ID')
# json_p = data[EXP_ID]['latency_profile']

# MBS = exp_config_parser.get_item('MBS', 0)
latency_profile = exp_config_parser.get_latency_profile()
# ref_times = exp_config_parser.get_ref_times()
# from dp import reorder_candidate_requests
torch.set_grad_enabled(False)

sys.path.insert(0, './')
assert os.environ["USE_TRT"] == "True"


# modules.USE_TRT = True


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
            data['restart'] = 'continuous'
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
        instance_index = int(params.pop('_independent_index'))
        self.module_name = f'UNetModule{instance_index}'
        self.sd_modules = sd_modules[self.module_name]

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
                data['restart'] = 'continuous'

            elif req["loop_index"]["UNetModule"] + 1 == req['loop_num']["UNetModule"]:
                data['restart'] = 'continuous'
            data['result'] = id

            # print(req)
        self.sd_modules.compute(reqs)

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

        # keys_to_remove = [k for k, v in req.items() if isinstance(v, torch.Tensor)]
        # for k in keys_to_remove:
        #     del req[k]
        for req in reqs:
            for k, v in req.items():
                if type(v) == torch.Tensor:
                    req[k] = None  # v.cpu()
                    # del req[k]
        # except Exception as e:
        #     print(f'error2 {e}', flush=True)
        # print(f'id v2: {id(req)}, {req_id} {req["id"]}')
        # print(f'vae time = {time.perf_counter()-start_time}')

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
    stage: int = -1
    left_time: float = 0
    running: bool = False
    ratio_hardware: float = 0.0
    # start_iteration_time: float = 0
    # stop_iteration_time: float = 0

    # defination of emergency:
    # profile_post = (finish_time - stop_iteration_time)
    # SLO_iter = SLO - (start_iteration_time - request_time) - profile_post
    # left_time = SLO_iter - (now - start_iteration_time)
    # left_round = total_round - current_round
    # emergency = left_round/left_time


class PyContinuousBatching:
    def init(self, params, options):
        print(f'init foom {params}')
        self.receiving_data = {}
        self.lock = threading.Lock()
        self.input_queue = queue.Queue()
        self._stop_event = threading.Event()

        self.cached_data = {}
        # self.target = omniback.get(params.pop('target'))
        self.target = {}
        self.target[0] = omniback.get('node.clip_backend')
        self.target[1] = omniback.get('node.unet_backend')
        self.target[2] = omniback.get('node.vae_backend')

        # self.max_batch_size = int(params.pop('max_batch_size', 21+0))
        slo_factor = params.pop('slo_factor')
        print(f'params.pop(slo_factor) = {slo_factor}')
        slo_factor = [float(x) for x in slo_factor.split(',')]
        if len(slo_factor) == 2:
            slo_factor = slo_factor[0]
        else:
            slo_factor = slo_factor[0]
        self.slo_factor = float(slo_factor)

        from helper import exp_config_parser
        self.stage_thres = [4, 5, 4]
        self.config_parser = exp_config_parser.ConfigParser(step_delta=0.95)
        self.latency_profile = exp_config_parser.get_latency_profile()
        self.max_batch_size = self.config_parser.get_max_batch_size(
            int(self.slo_factor*10))
        # self.scheduler = self.config_parser.get_scheduler()
        self.instance_num = 1
        # if self.scheduler.startswith('instance_'):
        #     self.instance_num = int(self.scheduler.split('_')[1])

        # self.waiting_ids = set()
        print(
            f'PyContinuousBatching max_batch_size = {self.max_batch_size} {self.config_parser.get_slo_factor_to_batch_size()} slo_factor= {self.slo_factor}')
        self.delayed = set()

        # self.latency_ratio = ((self.latency_profile['safety'][8]+self.latency_profile['vae'][8])/8)/(self.latency_profile['unet'][10]/10 )

        print(self.latency_profile)
        self.AVG_h = self.latency_profile['clip'][1]
        self.AVG_t = self.latency_profile['safety'][1]+self.latency_profile['vae'][1]
        self.AVG_i = self.latency_profile['unet'][1]
        
        self.AVG_h = (self.AVG_h, self.AVG_h)
        self.AVG_t = (self.AVG_t, self.AVG_t)
        self.AVG_i = (self.AVG_i, self.AVG_i)
        
        print(
            f'AVG HEADER LATENCY = [{self.AVG_h}, AVG ITERATION LATENCY = {self.AVG_i}, AVG TAIL LATENCY = {self.AVG_t}')
        # exit(0)
        self.active_ids = []


        self.thread = threading.Thread(target=self.task_loop, daemon=True)
        self.thread.start()

        self.running_ids = set()

    def get_latency(self, bs, stage=2):
        if stage == 1:
            return self.latency_profile['unet'][bs]
        elif stage == 2:
            return (self.latency_profile['safety'][bs]+self.latency_profile['vae'][bs])
        else:
            return self.latency_profile['clip'][bs]

    def get_avg_latency(self, bs, stage=2):
        if stage == 2:
            return (self.latency_profile['safety'][bs]+self.latency_profile['vae'][bs])/bs

    def get_last_stage_avg_latency(self, num_lst_stage):
        last_stage_avg_latency = []
        if num_lst_stage >= self.stage_thres[2]:
            last_stage_avg_latency += [self.AVG_t[0]] * \
                ((num_lst_stage / self.stage_thres[2]) * self.stage_thres[2])
            bs = num_lst_stage % self.stage_thres[2]
            if bs != 0:
                last_stage_avg_latency += [self.get_avg_latency(bs)] * (bs)
        return last_stage_avg_latency

    def all_received(self):
        return all(local_id in self.receiving_data for local_id in self.running_ids)

    def __del__(self):
        self._stop_event.set()
        self.thread.join(timeout=5.0)  # 超时保护
        if self.thread.is_alive():
            print("Warning: Worker thread did not exit cleanly")

    def forward(self, ios: List[omniback.Dict]):
        self.input_queue.put(('forward', ios))

    def task_loop(self):
        while not self._stop_event.is_set():
            # all in one thread !!!! no race condition
            try:
                step, task = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.on_start(task, step)

            #  if self.input_queue.empty():
            #     self.run_one_loop()

            self.input_queue.task_done()

    def on_start(self, ios, step):
        # print(f'on_start = {step}')
        if step == 'forward':
            for data in ios:
                # data['data'] is the request ID, and it's user's 责任去让他与实际数据一一对应
                self.receiving_data[data['data']] = data
        elif step == 'finish_stage':
            finished = self.on_finish_stage(ios)
            if len(finished) == 0:
                return
            for req_id in finished:
                del self.cached_data[req_id]
                self.running_ids.discard(req_id)
                print(f'request {req_id} finished')
        else:
            assert False, f'error step {step}'

        # print('receiving_data|running_ids: ', self.receiving_data.keys(), self.running_ids)
        if not self.input_queue.empty() or not self.all_received():
            return
        else:
            receiving_data = self.receiving_data
            self.receiving_data = {}
            self.running_ids = set()
        # self.num_stages = {i:0 for i in range(3)}
        # print('all data received: ', list(receiving_data.keys()))

        # start_scheduler_time = time.time()
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
                    data, ev, 0, loop_num=loop_num, time=req['start_iteration_time'], stage=0)
                # self.num_stages[0] += 1
            else:
                ev = data.pop('event')
                v = self.cached_data[req_id]
                if v.stage == 0:
                    v.stage += 1
                    v.loop_index = 0
                else:
                    v.loop_index += 1
                    if v.loop_index == v.loop_num:
                        v.stage += 1

                v.data = data
                v.event = ev
                # v.stage = stage
                v.loop_num = loop_num  # can be changed per iteration
                # self.num_stages[v.stage] += 1
                # print(f'on start id={req_id} stage={v.stage}, loop_index={v.loop_index}')

        # clear data
        # print(receiving_data, self.cached_data)
        receiving_data.clear()

        self.update_resource_status()
        if len(self.active_ids) == 0:
            return

        # activate_status = [(self.cached_data[x].stage, int(100*self.cached_data[x].ratio_hardware)) for x in active_ids]

        # print(f'activate ids: {active_ids} status={activate_status}')

        stage, most_emergent_ids, largest_status = self.get_most_emergent_ids(
            self.active_ids)
        # unactivate_ids = [x for x in active_ids if x not in most_emergent_ids]

        # self.active_ids = most_emergent_ids
        # print(f'decision => stage = {stage}, ids = {most_emergent_ids}/{unactivate_ids}, largest_status={largest_status}')
        self.running_ids = set(most_emergent_ids)

        self.run_stage(most_emergent_ids, stage)

        # all_time = time.time() - start_scheduler_time
        # print(f'scheduler time = {all_time}, cached = {len(self.cached_data)}, running = {len(self.running_ids)}, active_ids = {len(active_ids)}')
        # # exit(0)

    def get_most_emergent_ids(self, active_ids):
        largest_id = max(
            active_ids, key=lambda x: self.cached_data[x].ratio_hardware)
        stage = self.cached_data[largest_id].stage
        left_time = self.cached_data[largest_id].left_time
        ori_active_ids = [
            x for x in active_ids if self.cached_data[x].stage == stage]
        ori_active_ids.sort(
            key=lambda x: self.cached_data[x].ratio_hardware, reverse=True)
        # bs = 1
        # while (self.get_latency(bs +1, stage) < left_time) and bs +1 <= len(active_ids):
        #     bs += 1
        hr = self.cached_data[largest_id].ratio_hardware
        bs = min(int(1/hr), len(ori_active_ids))
        # bs = min(self.max_batch_size, len(ori_active_ids))
        # if bs >10:
        #     bs = int(bs/2)
        # elif bs > 10:
        #     bs = int(bs/2 + 1)
        # elif bs >= 8:
        #     bs = 7
        # bs = min(bs, self.stage_thres[stage] * 2)
        active_ids = ori_active_ids[:bs]
        # print(f'active_ids = {active_ids},stage={stage} hr={hr:.3f} act={bs}/{len(ori_active_ids)} total={len(self.cached_data)}')
        # (largest_id, self.cached_data[largest_id].ratio_hardware, left_time, bs)
        return stage, active_ids, None

    def update_resource_status(self):
        need_drop = set()

        candidate_requests = []

        # num_stage_last = []
        num_stage_last = sum(
            1 for x in self.cached_data.values() if x.stage == 2)

        for req_id, v in self.cached_data.items():  # 修改循环变量为v
            req = global_request_pool[req_id]
            SLO = req['SLO']
            request_time = req['request_time']
            now = time.time()
            v.left_time = max(1e-6, request_time + SLO - now)

            if v.stage == 0:
                v.resource = (
                    self.AVG_h[1] + self.AVG_t[1] + v.loop_num * self.AVG_i[0])

                if (self.AVG_h[1] + self.AVG_t[1] + (v.loop_num) * self.AVG_i[1]) > v.left_time:
                    v.resource += v.left_time
            elif v.stage == 1:
                v.resource = (
                    self.AVG_t[1] + (v.loop_num - v.loop_index) * self.AVG_i[0])
                # if v.resource / v.left_time > 0.2:
                #     v.resource += v.left_time * 3
                # if (v.loop_num - v.loop_index) * self.AVG_i[0]/ (v.left_time - self.AVG_t[1]) > 0.2:
                #     resource += v.left_time * 3
                if (self.AVG_t[1] + (v.loop_num - v.loop_index) * self.AVG_i[1]) > v.left_time:
                    v.resource += v.left_time*2
                # if resource > v.left_time * 0.5:
                #     resource = v.left_time * 1.5
            elif v.stage == 2:
                v.resource = self.AVG_t[1]
            else:
                assert False, v.stage

            v.ratio_hardware = v.resource / v.left_time

            if v.ratio_hardware >= 1:
                need_drop.add(req_id)
            else:
                if v.stage != 2:
                    candidate_requests.append(
                        (req_id, v.resource, v.left_time))
                elif num_stage_last > 0:
                    # 修正最后一阶段的资源占用情况
                    v.resource = (self.latency_profile['safety'][num_stage_last] +
                                  self.latency_profile['vae'][num_stage_last])/num_stage_last
                    # v.ratio_hardware = resource / v.left_time
                    # resource += 1e-6 *(1- v.ratio_hardware)
                    candidate_requests.append(
                        (req_id, v.resource, v.left_time))
        # drop2, candidate_requests = reorder_candidate_requests(candidate_requests)

        # print(f'candidate_requests = {candidate_requests}')

        # last_stage_avg_latency = self.get_last_stage_avg_latency(len(num_stage_last))

        # need_drop = need_drop.union(drop2)

        if len(need_drop) > 0:
            # print(f'drop2 = {drop2}')
            self.drop_request(need_drop)

        self.active_ids = [x for x in self.active_ids if x in self.cached_data]
        sum_resource = sum(
            [self.cached_data[x].resource for x in self.active_ids])
        if sum_resource >= 1:
            # candidate = self.active_ids
            pass
        else:
            candidate_requests = [
                x[0] for x in candidate_requests if x[0] not in self.active_ids]
            # candidate_requests.sort(key=lambda x: (self.cached_data[x].resource))
            candidate_requests.sort(key=lambda x: (
                self.cached_data[x].delay, self.cached_data[x].time))
            # z = [ (x, self.cached_data[x].delay, self.cached_data[x].resource) for x in candidate_requests]
            # if len(z) > 0:
            #     print(z)
            self.active_ids += candidate_requests
        # sum_rsc =
        # for req_id in self.active_ids:

        cumulative_ratio = 0
        active_ids = []  # {i:[] for i in range(3)}
        for req_id in self.active_ids:
            v = self.cached_data[req_id]
            if cumulative_ratio + v.ratio_hardware < 1.0:
                cumulative_ratio += v.ratio_hardware
                active_ids.append(req_id)
        self.active_ids = active_ids

        for k, v in self.cached_data.items():
            if k not in self.active_ids:
                v.delay += 1
                # active_ids[v.stage].append(req_id)
        # print([(x, self.cached_data[x].ratio_hardware) for x in self.active_ids])
        # 总的资源量越小越好

    def run_stage(self, ids, target):
        ios = []
        for k in ids:
            v = self.cached_data[k]
            ios.append(v.data)
            assert 'event' not in v.data

        ev = omniback.Event(len(ios))
        ev.set_callback(lambda: self.input_queue.put(('finish_stage', ids)))
        ev.set_exception_callback(lambda x: print(
            f'error {x} in Stage {target} '))
        for item in ios:
            item['event'] = ev
        # print(f'cb forward {len(io)}')
        self.target[target].forward(ios, None)

    def on_finish_stage(self, ids):
        should_del = []
        for k in ids:
            v = self.cached_data[k]
            if v.stage == 0 or v.stage == 1:
                # v.stage = 1
                # print(f'request {k} in stage {v.stage} finished')
                pass

                # if v.loop_index + 1 >=v.loop_num:
                #     v.stage += 1
                # req['stop_iteration_time'] = stop_iteration_time
                # req['stop_iteration_batch'] = len(self.cached_data)
            elif v.stage == 2:
                should_del.append(k)

            v.data['event'] = v.event
            v.event.notify_all()
            # print(f'request {k} notified')

        return should_del
    # def on_finish(self, ids):
    #     should_del = []
    #     for k in ids :
    #         v=  self.cached_data[k]
    #         if v.loop_index + 1 >=v.loop_num:
    #             should_del.append(k)
    #             req = global_request_pool[k]
    #             # req['stop_iteration_time'] = stop_iteration_time
    #             # req['stop_iteration_batch'] = len(self.cached_data)
    #         v.data['event'] = v.event
    #         v.event.notify_all()
    #             # print(f'request {k} notified')
    #     for req_id in should_del:
    #         del self.cached_data[req_id]
    #         print(f'request {req_id} finished')
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
            left_num = v.loop_num - v.loop_index
            print(
                f'drop req. {req_id}. hardware ratio = {v.ratio_hardware} left_time = {v.left_time}, left_num={left_num}')

            global_request_pool[req_id]['loop_index']['UNetModule'] = -1

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
    omniback.pipe({'unet_backend': {'backend': 'SyncTensor[Unet]', 'batching_timeout': 0, 'instance_num': 1},
               'clip_backend': {'backend': 'SyncTensor[Clip]', 'batching_timeout': 8, 'instance_num': 1},
               'vae_backend': {"backend": 'SyncTensor[Vae]', 'batching_timeout': 8, 'instance_num': 1}
               })

    config = {
        'global': {'entrypoint': 'Restart[DagDispatcher]'},
        # 'clip':{'backend':'SyncTensor[Clip]','batching_timeout':8,'instance_num':1,'next':'unet'},  #
        'continuous': {'node_entrypoint': 'Register[PyContinuousBatching]', 'slo_factor': slo_factor}}

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
