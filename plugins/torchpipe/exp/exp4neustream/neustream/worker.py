import os
import time
import queue
import torch
import torch.multiprocessing as multiprocessing
import threading
import json
from datetime import datetime, timezone, timedelta

# PriorityQueue can not handle same value case, so need this class


class ComparableRequest:
    def __init__(self, priority, data):
        self.priority = priority
        self.data = data

    def __lt__(self, other):
        # Define comparison based on priority or any other logic
        return self.priority < other.priority


class Worker(multiprocessing.Process):
    # class Worker(threading.Thread):
    def __init__(self, stream_module_list, input_queue, output_queue, id, log_prefix, deploy_ready, extra_vae_safety_time, image_size, profile_device, step_slo_scale: float, step_delta: float, **kwargs):
        super().__init__()
        self.stream_module_list = stream_module_list
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.id = id  # mark worker kind
        self.batch_queue = queue.PriorityQueue()  # (priority, item). lowest first
        self.low_priority_batch_queue = queue.PriorityQueue()
        self.current_batch = []
        self.batch_ready = threading.Semaphore(0)
        self.batch_used = threading.Semaphore(0)
        self.first_batch = True
        self.loop_module_list = [type(
            stream_module).__name__ for stream_module in self.stream_module_list if stream_module.loop_module]
        module_name_list = [
            type(stream_module).__name__ for stream_module in self.stream_module_list]
        self.module_tag = "&".join(module_name_list)
        self.loop_unit = 1 if len(self.loop_module_list) != 0 else 1

        self.batch_upper_bound = 0
        if profile_device == "h100":
            if image_size == 256:
                self.batch_upper_bound = 40
            elif image_size == 512:
                self.batch_upper_bound = 15
        elif profile_device == "rtx4090":
            if image_size == 256:
                self.batch_upper_bound = 40
            elif image_size == 512:
                self.batch_upper_bound = 15
        latency_profile = kwargs.get("latency_profile", None)
        self.latency_profile = latency_profile

        self.instant_profile = {}
        self.instant_profile_trust = 999999  # an impossible value
        self.deploy_ready = deploy_ready
        self.extra_vae_safety_time = extra_vae_safety_time
        for batch_size in range(self.batch_upper_bound+1):
            self.instant_profile[batch_size] = {}
            self.instant_profile[batch_size]["count"] = 0
            self.instant_profile[batch_size]["average_latency"] = 0

        self.total_request_count = 0
        self.goodput = 0
        self.log_prefix = log_prefix
        if "device" in kwargs:
            self.device = kwargs["device"]
            for stream_module in self.stream_module_list:
                stream_module.device = self.device
        else:
            self.device = "cuda"  # default to cuda:0

        self.step_slo_scale = step_slo_scale
        self.step_delta = step_delta
        # one step slo, controlled by two values
        self.slo_batch_size = 0
        print(f"yhc debug:: step_factor = {step_slo_scale * step_delta}")

        if self.id == "clip":
            self.step_slo_latency = self.latency_profile["clip"][1] * \
                self.step_slo_scale * self.step_delta
            while self.latency_profile["clip"][(self.slo_batch_size + 1)] < self.step_slo_latency:
                self.slo_batch_size += 1
                if self.slo_batch_size >= self.batch_upper_bound:
                    break
        elif self.id == "unet":
            self.step_slo_latency = self.latency_profile["unet"][1] * \
                self.step_slo_scale * self.step_delta
            while self.latency_profile["unet"][(self.slo_batch_size + 1)] < self.step_slo_latency:
                self.slo_batch_size += 1
                if self.slo_batch_size >= self.batch_upper_bound:
                    break
        elif self.id == "vae&safety":
            self.step_slo_latency = (
                self.latency_profile["vae"][1] + self.latency_profile["safety"][1]) * self.step_slo_scale * self.step_delta
            while (self.latency_profile["vae"][(self.slo_batch_size + 1)] + self.latency_profile["safety"][(self.slo_batch_size + 1)]) < self.step_slo_latency:
                self.slo_batch_size += 1
                if self.slo_batch_size >= self.batch_upper_bound:
                    break
        print(
            f"yhc debug:: hold module:{self.module_tag}, slo_batch_size={self.slo_batch_size}")
        print(
            "-"*10, f"yhc debug:: loop_module_list: {self.loop_module_list}", "-"*10)

    def set_device(self, device, **kwargs):
        self.device = device

    def deploy(self, **kwargs):
        # deploy all stream_modules
        for stream_module in self.stream_module_list:
            stream_module.deploy()

    def gather(self, **kwargs):
        # determine whether terminate the process,
        self.terminate_receive_flag = False
        self.terminate_schedule_flag = False
        while True:
            # put request to batch_queue
            while not self.input_queue.empty():
                request = self.input_queue.get()
                if request == None:
                    print(
                        f"pid: [{os.getpid()}], holding module: {self.module_tag}, received terminate signal!")
                    self.output_queue.put(None)
                    self.terminate_receive_flag = True
                    # if self.id == "unet":
                    #     batch_latency_record = open(
                    #         f"profiles/unet_batch_latency.log", "w")
                    #     json.dump(self.instant_profile, batch_latency_record)
                    break

                # log the request info
                if self.id == "unet":
                    info = {
                        "request_time": request["request_time"],
                        "id": request["id"],
                        "SLO": request["SLO"]
                    }
                    # self.log_file.write(json.dumps(info)+"\n")

                self.total_request_count += 1
                request[self.module_tag+"_receive_time"] = time.time()
                # check whether can serve
                if self.id == "clip":
                    if request["SLO"] + request["request_time"] < time.time() + self.latency_profile["clip"][1] + self.latency_profile["unet"][1] * request["loop_num"]["UNetModule"] + self.latency_profile["vae"][1] + self.latency_profile["safety"][1]:
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{request['id']}"
                        # self.log_file.write(info+"\n")
                        print(info)
                    else:
                        # check tensor whether on same device
                        for key in request.keys():
                            if type(request[key]) == torch.Tensor:
                                request[key] = request[key].to(self.device)
                        request["wait_loop_count"] = 0
                        self.batch_queue.put(
                            ComparableRequest(time.time(), request))
                elif self.id == "unet":
                    if request["SLO"] + request["request_time"] < time.time() + self.latency_profile["unet"][1] * request["loop_num"]["UNetModule"] + self.latency_profile["vae"][1] + self.latency_profile["safety"][1]:
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{request['id']}"
                        # self.log_file.write(info+"\n")
                        print(info)
                    else:
                        for key in request.keys():
                            if type(request[key]) == torch.Tensor:
                                request[key] = request[key].to(self.device)
                        request["wait_loop_count"] = 0
                        self.batch_queue.put(
                            ComparableRequest(time.time(), request))
                elif self.id == "vae&safety":
                    if request["SLO"] + request["request_time"] < time.time() + self.latency_profile["vae"][1] + self.latency_profile["safety"][1]:
                        info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{request['id']}"
                        # self.log_file.write(info+"\n")
                        print(info)
                        print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, goodput rate: {self.goodput / self.total_request_count}, goodput: {self.goodput}, total_request: {self.total_request_count}")
                    else:
                        # check tensor whether on same device
                        for key in request.keys():
                            if type(request[key]) == torch.Tensor:
                                request[key] = request[key].to(self.device)
                        request["wait_loop_count"] = 0
                        self.batch_queue.put(
                            ComparableRequest(time.time(), request))
                # self.log_file.flush()

            # stop working
            if self.batch_queue.qsize() == 0 and self.terminate_receive_flag and len(self.current_batch) == 0:
                self.terminate_schedule_flag = True
                # release lock, let running loop liberate from lock
                self.batch_ready.release()
                print(
                    f"pid: [{os.getpid()}], holding module: {self.module_tag}, terminate schedule!")
                break

            # empty queue, no need to form batch
            if self.batch_queue.qsize() == 0 and len(self.current_batch) == 0:
                continue

            # avoid concurrent access of self.current_batch
            if self.first_batch:
                self.first_batch = False
            else:
                self.batch_used.acquire()
                # print(f"{self.module_tag}, batch_used acquired once.")
            self.schedule_begin = time.time()

            def scatter():
                # put finished request to output_queue
                if len(self.current_batch) != 0:
                    if len(self.loop_module_list) != 0:
                        for loop_module in self.loop_module_list:
                            for item in self.current_batch:
                                if item.data["loop_index"][loop_module] >= item.data["loop_num"][loop_module]:
                                    item.data[self.module_tag +
                                              "_send_time"] = time.time()
                                    for key in item.data.keys():
                                        if type(item.data[key]) == torch.Tensor:
                                            item.data[key] = item.data[key].cpu()
                                    self.output_queue.put(item.data)
                                else:
                                    # 没跑够loop_count，重新回queue
                                    self.batch_queue.put(item)
                    # no loop module
                    else:
                        # self.total_request_count += len(self.current_batch)
                        for item in self.current_batch:
                            item.data[self.module_tag +
                                      "_send_time"] = time.time()
                            if time.time() <= item.data["SLO"] + item.data["request_time"]:
                                self.goodput += 1
                            # tensor have to be moved to cpu when passing across device
                                for key in item.data.keys():
                                    if type(item.data[key]) == torch.Tensor:
                                        item.data[key] = item.data[key].cpu()
                                self.output_queue.put(item.data)
                        info = f"goodput: {self.goodput}, total_request: {self.total_request_count}\n"
                        # self.log_file.write(info)
                        print(f"pid: [{os.getpid()}], holding module: {self.module_tag}, goodput rate: {self.goodput / self.total_request_count}, goodput: {self.goodput}, total_request: {self.total_request_count}")
                # self.log_file.flush()
            scatter()

            # 将batch_queue里所有超时的request都扔掉
            if self.id == "unet":
                valid_request_list = []
                while not self.batch_queue.empty():
                    item = self.batch_queue.get()
                    # 超时则丢弃
                    # 尝试用过去run的batch信息来profile
                    if self.instant_profile[1]["count"] >= self.instant_profile_trust:
                        if item.data["request_time"] + item.data["SLO"] <= time.time() + self.instant_profile[1]["average_latency"] * (item.data["loop_num"]["UNetModule"] - item.data["loop_index"]["UNetModule"]) + self.latency_profile["vae"][1] + self.latency_profile["safety"][1]:
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            # self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                    # instant_profile次数比较少，不采纳
                    else:
                        if item.data["request_time"] + item.data["SLO"] <= time.time() + self.latency_profile["unet"][1] * (item.data["loop_num"]["UNetModule"] - item.data["loop_index"]["UNetModule"]) + self.latency_profile["vae"][1] + self.latency_profile["safety"][1]:
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            # self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                # 有效的request放回batch_queue
                for item in valid_request_list:
                    # 更新priority
                    # 不更新priority，避免出现交替schedule
                    # new_rest_time = item[1]["request_time"] + item[1]["SLO"] - time.time()
                    self.batch_queue.put(item)

                valid_request_list = []
                while not self.low_priority_batch_queue.empty():
                    item = self.low_priority_batch_queue.get()
                    # 超时则丢弃
                    # 尝试用过去run的batch信息来profile
                    if self.instant_profile[1]["count"] >= self.instant_profile_trust:
                        if item.data["request_time"] + item.data["SLO"] <= time.time() + self.instant_profile[1]["average_latency"] * (item.data["loop_num"]["UNetModule"] - item.data["loop_index"]["UNetModule"]) + self.latency_profile["vae"][1] + self.latency_profile["safety"][1]:
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            # self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                    # instant_profile次数比较少，不采纳
                    else:
                        if item.data["request_time"] + item.data["SLO"] <= time.time() + self.latency_profile["unet"][1] * (item.data["loop_num"]["UNetModule"] - item.data["loop_index"]["UNetModule"]) + self.latency_profile["vae"][1] + self.latency_profile["safety"][1]:
                            info = f"pid: [{os.getpid()}], holding module: {self.module_tag}, abandon one request. id:{item.data['id']}"
                            # self.log_file.write(info+"\n")
                            print(info)
                        else:
                            valid_request_list.append(item)
                # 有效的request放回batch_queue
                for item in valid_request_list:
                    # 更新priority
                    # 不更新priority，避免出现交替schedule
                    self.low_priority_batch_queue.put(item)
            # 及时写回
            # self.log_file.flush()

            # form new batch
            self.current_batch = []

            self.high_priority_count = 0
            self.low_priority_count = 0

            # first put request with no wait_loop_count
            while self.batch_queue.qsize() > 0 and len(self.current_batch) < self.slo_batch_size:
                self.high_priority_count += 1
                self.current_batch.append(self.batch_queue.get())

            # put unscheduled request into low_priority queue
            while not self.batch_queue.empty():
                new_item = self.batch_queue.get()
                new_item.data["wait_loop_count"] += 1
                self.low_priority_batch_queue.put(ComparableRequest(
                    new_item.data["wait_loop_count"], new_item.data))

            # if not exceed slo_batch_size, add low priority request
            while self.low_priority_batch_queue.qsize() > 0 and len(self.current_batch) < self.slo_batch_size:
                self.low_priority_count += 1
                self.current_batch.append(self.low_priority_batch_queue.get())

            # update the wait_loop_count in low_priority_queue
            temp_request_list = []
            while not self.low_priority_batch_queue.empty():
                request = self.low_priority_batch_queue.get().data
                request["wait_loop_count"] += 1
                temp_request_list.append(request)
            for request in temp_request_list:
                self.low_priority_batch_queue.put(
                    ComparableRequest(request["wait_loop_count"], request))

            self.schedule_end = time.time()
            self.batch_ready.release()
            # self.log_file.write(f"")
            # self.log_file.flush()

    @torch.no_grad()
    def run(self, **kwargs):
        # log_file = open(f"{self.log_prefix}_{self.module_tag}.log", "w")
        # self.log_file = log_file
        torch.set_grad_enabled(False)
        print(f'device={self.device}')
        torch.cuda.set_device(self.device)
        with torch.inference_mode():
            try:
                # print("yhc test run.")
                print(
                    f"pid: [{os.getpid()}], module list: {self.stream_module_list}")
                for module in self.stream_module_list:
                    module.device = self.device
                    module.deploy()
                    print(
                        f"pid: [{os.getpid()}], serving module: {type(module).__name__}")
                self.deploy_ready.release()

                schedule_batch_thread = threading.Thread(target=self.gather)
                schedule_batch_thread.start()

                # sequentially run the batch module
                while True:
                    if self.terminate_schedule_flag == True:
                        print(
                            f"pid: [{os.getpid()}], holding module: {self.module_tag}, terminate running!")
                        break
                    self.batch_ready.acquire()
                    # print(f"{self.module_tag}, batch_ready acquired once.")
                    if len(self.current_batch) == 0:
                        info = {
                            "batch_size": 0,
                            "time": time.time(),
                            "queue_size_before_schedule": 0,
                            "msg": "emptyqueue"
                        }
                        # log_file.write(json.dumps(info) + "\n")
                        self.batch_used.release()
                        continue
                    # write the batch log

                    execution_begin = time.perf_counter()

                    # wipe off the priority
                    # below list just passed reference, not data copy
                    batch_request = [item.data for item in self.current_batch]
                    # execute through the pipeline
                    if self.id == "unet":
                        for _ in range(1):
                            batch_request = self.stream_module_list[0].compute(
                                batch_request)
                            # update loop_index
                            for request in batch_request:
                                request["loop_index"][self.loop_module_list[0]] += 1
                    else:
                        for module in self.stream_module_list:
                            batch_request = module.compute(batch_request)
                    info = {
                        "batch_size": len(self.current_batch),
                        "time": time.time(),
                        "schedule_time": self.schedule_end - self.schedule_begin,
                        "execution_time": time.perf_counter() - execution_begin,
                        "high_priority_count": self.high_priority_count,
                        "low_priority_count": self.low_priority_count,
                        "queue_size_before_schedule": self.batch_queue.qsize() + len(self.current_batch) + self.low_priority_batch_queue.qsize(),
                        "batch_size_after_schedule": self.batch_queue.qsize(),
                        "running_requests_id_list": [item.data["id"] for item in self.current_batch],
                        "rest_time": [(item.data["request_time"] + item.data["SLO"] - time.time()) for item in self.current_batch]
                    }
                    # log_file.write(json.dumps(info) + "\n")
                    # calculate the new batch latency
                    if self.id == "unet":
                        latency_profile = time.perf_counter() - execution_begin
                        current_batch_size = len(self.current_batch)

                        new_average_latency = (self.instant_profile[current_batch_size]["average_latency"] * self.instant_profile[current_batch_size]
                                               ["count"] + latency_profile) / (self.instant_profile[current_batch_size]["count"] + 1)
                        self.instant_profile[current_batch_size]["average_latency"] = new_average_latency
                        self.instant_profile[current_batch_size]["count"] += 1
                        # self.log_file.write(json.dumps(self.instant_profile)+"\n")"""
                    self.batch_used.release()

            # Worker process receive interrupt
            except KeyboardInterrupt:
                print(
                    "-"*10, f"Worker process:[{os.getpid()}] received KeyboardInterrupt.", "-"*10)
