import uuid
import time
import torch
import json, sys, os
import multiprocessing

# from stable_diffusion_v1_5.stable_diffusion_pipeline import StableDiffusionPipeline
# from stable_diffusion_v1_5.stable_diffusion_scheduler import StableDiffusionScheduler
# from utils import *
from helper import exp_config_parser
latency_profile = exp_config_parser.get_latency_profile()

ref_times = [latency_profile['clip'][1],
             latency_profile['unet'][1], latency_profile['vae'][1]+latency_profile['safety'][1]]

def handle_output(output_queue, log_name, workload_request_count, rate, cv, slo_factor, trace_time):

    from datetime import datetime, timezone, timedelta
    # timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")

    goodput_request_count = 0 # not include warm-up request
    request_count = 0
    
    # f = open(log_name+timestamp, "w")

    output_queue.get()
    request_count += 1
    total_start_time = None

    while True:
        if output_queue.qsize() == 0:
            time.sleep(0.01)
            continue
        result = output_queue.get()
        
        if total_start_time is None and result["id"] >= 0:
            total_start_time = result['request_time']
            
        # stop the process
        if result == None:
            total_trace_time = time.time() - total_start_time
            # f.close()
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            statistics = f"time:{formatted_time}, id:neustream_sd_256, rate:{rate} qps, cv={cv}, slo={slo_factor}, NeuStream goodput_rate={goodput_request_count}/{workload_request_count}, goodput speed={goodput_request_count/total_trace_time}\n"
            print(statistics)
            result_file = open("stable_diffusion_serve_result.txt", "a")
            result_file.write(statistics)
            break
        request_count += 1
        print(f"Collector handle request_count: {request_count}")
        # warm-up request
        if result["id"] < 0:
            continue

        goodput_request_count += 1
        #print("-"*20)
        # f.write("-"*20+"\n")
        result["finish_time"] = time.time()
        # f.write(f"Server-side end2end latency: {result['finish_time'] - result['request_time']}\n")
        print(f"Server-side end2end latency: {result['finish_time'] - result['request_time']}")
        # f.write(f'request step: {result["loop_num"]["UNetModule"]}')
        computation_time = 0
        for key in result.keys():
            if "receive_time" in key:
                computation_time -= result[key]
            if "send_time" in key:
                computation_time += result[key]
            # if type(result[key]) == float or key == "id" or type(result[key]) == int:
            #     f.write(f"key: {key}, value: {result[key]}\n")
                #print(f"key: {key}, value: {result[key]}")
        transmission_time = result['finish_time'] - result['request_time'] - computation_time
        # f.write(f'intra module time = {computation_time}\n')
        # f.write(f'transmission time = {transmission_time}\n')
        #img = PIL.Image.fromarray(numpy.array(result["pillow_image"]).astype(numpy.uint8))
        #img.save(f"{result['id']}.jpg")
        #print("-"*20)
        # f.write(f"collector worker receive workload request count = {goodput_request_count}\n")
        print(f"collector worker receive workload request count = {goodput_request_count}")
        # f.flush()

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="parameter for server.")

    parser.add_argument('--image_size', required=True, type=int, help='a value to determine image size')
    parser.add_argument('--rate_scale', required=True, type=str, help='a value to determine arrival rate')
    parser.add_argument('--cv_scale', required=True, type=str, help='a value to determine arrival coefficient of variation')
    parser.add_argument('--slo_scale', required=True, type=str, help='a value to determine slo factor')
    parser.add_argument('--extra_vae_safety_time', required=True, type=float, help='a value to determine extra budget for vae and safety')
    parser.add_argument('--log_folder', required=True, type=str, help='a value to determine log folder')
    parser.add_argument('--profile_device', required=False, default='rtx4090',
                        type=str, help='a value to determine profile device')
    parser.add_argument('--step_delta', required=True, type=float, help='a value to determine running device')

    args = parser.parse_args()

    key = f"rate={args.rate_scale}_cv={args.cv_scale}_{args.image_size}"

    rate = float(args.rate_scale)

    slo_factor = float(args.slo_scale)

    cv = float(args.cv_scale)

    image_size = args.image_size
    
    extra_vae_safety_time = args.extra_vae_safety_time

    log_folder = args.log_folder

    step_delta = args.step_delta
    
    trace = json.load(open("data/SD_FP16_img256_trace.json"))
    arrival_interval_list = trace[f"rate={float(args.rate_scale):.2f},cv={float(args.cv_scale):.2f}"]
    random_step_list = trace["random_step_list"]
    
    module_latency_variable = latency_profile
    
    # no need to pass gradient
    torch.set_grad_enabled(False)
    try:
        # init pipeline from config
        sd_config_file = "stable_diffusion_v1_5/config.json"
        assert os.environ["USE_TRT"] == "True"

        class StableDiffusionPipeline():
            def __init__(self, config_path, **kwargs):
                super().__init__()
                sys.path.insert(0, './')
                from trt_modules import ClipModule, UNetModule, VaeModule, SafetyModule

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

                    self.modules[key] = self.modules_type[key](**value)
                self.stream_module_list = [self.modules['ClipModule'],
                                           self.modules['UNetModule'],
                                             self.modules['VaeModule'],
                                             self.modules['SafetyModule']]
            def default_deploy(self, **kwargs):
                import omniback
                import torchpipe
                # for module in self.stream_module_list:
                for k, v in self.modules.items():
                    v.deploy()


        # 初始化管道
        sd_config_file = "stable_diffusion_v1_5/config.json"
        sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)        

        # trace setting
        from neustream.test_set import prompt_list

        #time_pattern = "uniform"
        time_pattern = "request=500"
        workload_request_count = 500

        #delay_num = args.delay_num
        from datetime import datetime, timezone, timedelta
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        log_prefix = f"{log_folder}/{timestamp}_image_size={image_size}_{time_pattern}_rate={rate}_cv={cv}_slo_factor={slo_factor}_extra_vae_time={extra_vae_safety_time}_step_delta={step_delta}"

        # init queue
        worker_nums = 3
    
        input_queue = torch.multiprocessing.Manager().Queue()
        output_queue = torch.multiprocessing.Manager().Queue()
        queue_list = [torch.multiprocessing.Manager().Queue() for _ in range(worker_nums - 1)]

        deploy_ready = torch.multiprocessing.Semaphore(0)
        
        queue_list.insert(0, input_queue)
        queue_list.append(output_queue)        

        device = "cuda:0"
        # 创建工作进程
        print(torch.multiprocessing.get_start_method())
        from neustream.worker import Worker
        worker_list = []
        worker_list.append(Worker(stream_module_list=sd_pipeline.stream_module_list[0:1], input_queue=queue_list[0], output_queue=queue_list[1], id="clip", log_prefix=log_prefix, deploy_ready=deploy_ready,
                           extra_vae_safety_time=args.extra_vae_safety_time, image_size=image_size, profile_device=args.profile_device, device=device, step_slo_scale=slo_factor,  step_delta=step_delta, latency_profile=latency_profile))
        worker_list.append(Worker(stream_module_list=sd_pipeline.stream_module_list[1:2], input_queue=queue_list[1], output_queue=queue_list[2], id="unet", log_prefix=log_prefix, deploy_ready=deploy_ready,
                           extra_vae_safety_time=args.extra_vae_safety_time, image_size=image_size, profile_device=args.profile_device, device=device, step_slo_scale=slo_factor,  step_delta=step_delta, latency_profile=latency_profile))
        worker_list.append(Worker(stream_module_list=sd_pipeline.stream_module_list[2:], input_queue=queue_list[2], output_queue=queue_list[3], id="vae&safety", log_prefix=log_prefix, deploy_ready=deploy_ready,
                           extra_vae_safety_time=args.extra_vae_safety_time, image_size=image_size, profile_device=args.profile_device, device=device, step_slo_scale=slo_factor,  step_delta=step_delta, latency_profile=latency_profile))

        # trace_time = sum(arrival_interval_list[:workload_request_count])/1.2 # ? why 1.2
        trace_time = sum(arrival_interval_list[:workload_request_count])
        collect_worker = torch.multiprocessing.Process(target=handle_output, args=(output_queue, log_prefix, workload_request_count, rate, cv, slo_factor, trace_time))
        collect_worker.start()

        deploy_begin = time.time()
        for _worker in worker_list:
            _worker.start()
        
        # wait for all module ready
        for _ in range(len(worker_list)):
            deploy_ready.acquire()
        print(f"Workers deploy all done! time used: {time.time() - deploy_begin}")
        
        warm_up_request_temp = {
            "prompt": "a beautiful girl studying in Chinese University",
            "height": image_size,
            "width": image_size,
            "loop_num": {
                "UNetModule": 50
            },
            "request_time": time.time(),
            "guidance_scale": 7.5,
            "seed": 0,
            "SLO": 10000,
            "loop_index": {
                "UNetModule": 0
            },
            "id": -1
        }
        # warm up all the unet batch_size
        test_count = 30
        for idx in range(test_count):
            #warm_up_request["id"] = idx
            warm_up_request = warm_up_request_temp.copy()
            time.sleep(0.08)
            warm_up_request["request_time"] = time.time()
            input_queue.put(warm_up_request)
        # warm up all the unet batch_size
        
        # sleep until warm-up end
        time.sleep(20)
        print("warm up succeed!")

        #steps_range = list(range(-5, 6))

        for idx in range(workload_request_count):
            time.sleep(arrival_interval_list[idx])
            input = {
                "prompt": prompt_list[idx%100],
                "height": image_size,
                "width": image_size,
                "loop_num": {
                    "UNetModule": random_step_list[idx],#50 + steps_range[idx % 11]#steps_list[idx%100]
                },
                "loop_index": {
                    "UNetModule": 0
                },
                "guidance_scale": 7.5,
                "seed": 0,
                "uuid": uuid.uuid1(),
                "request_time": time.time(),
                "SLO": slo_factor * (module_latency_variable["clip"][1] + module_latency_variable["unet"][1] * random_step_list[idx] + module_latency_variable["vae"][1] + module_latency_variable["safety"][1]),
                "id": idx # for debug
            }
            input_queue.put(input)
            # time.sleep(5)
            print(f"clip_queue put item: {input['id']}\n-----------------------")
        # end request
        time.sleep(10)
        input_queue.put(None)
        
        for _worker in worker_list:
            _worker.join()

    except KeyboardInterrupt:
        print("-"*10, "Main process received KeyboardInterrupt","-"*10)