import uuid
import time
import torch
import json
import os
import pickle
import omniback


# max_batch_size = 1

from helper import exp_config_parser

latency_profile = exp_config_parser.get_latency_profile()

ref_times = [latency_profile['clip'][1],
             latency_profile['unet'][1], latency_profile['vae'][1]+latency_profile['safety'][1]]

workload_request_count = 500


def handle_output(output_queue, log_name, workload_request_count, rate, cv, slo_scale):

    from datetime import datetime, timezone, timedelta

    goodput_request_count = 0  # not include warm-up request
    drop_count = 0
    request_count = 0

    total_start_time = None
    while True:
        # if output_queue.qsize() == 0:
        #     time.sleep(0.01)
        #     continue
        result = output_queue.get()

        if total_start_time is None and result["id"] >= 0:
            total_start_time = result['request_time']

        request_count += 1

        # print(result.keys(), 'ooo')
        # freeness.release()

        if result == None:

            total_trace_time = time.time() - total_start_time

            goodput_request_count -= drop_count

            # f.close()
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

            slos = 0#[slo2bs_bable[int(x*10)] for x in slo_scale]
            statistics = f"time:{formatted_time}, id:omniback_sd_256_candrop{can_drop}, rate:{rate} qps, cv={cv}, goodput_rate={goodput_request_count}/{workload_request_count}, goodput speed={goodput_request_count/total_trace_time}, dropped: {drop_count} slo_scale={slo_scale}, mbs={slos}\n"
            print(statistics)

            result_file = open("stable_diffusion_serve_result.txt", "a")
            result_file.write(statistics)
            break
        if "exception" in result:
            print("error in callback")
            raise RuntimeError(result['exception'])

        # print(f"Collector handle request_count: {request_count}, {result["id"]}")

        finish_time = time.time()
        # local_free = result.pop('freeness', None)
        # if local_free:

        request_time = result['request_time']
        # / {result['cb_time'] -  result['request_time']}
        print(
            f"(id={result['id']}) Server-side end2end latency: {finish_time - result['request_time']}")

        if result["id"] < 0:
            # warm-up request
            # from PIL import Image
            # img = result['pillow_image']
            # img.save("output_image.jpg", quality=95)
            # print('saved to output_image.jpg')
            continue

        slo = result['SLO']
        if finish_time - request_time <= slo:
            goodput_request_count += 1
            loop_index = result['loop_index']['UNetModule']
            if loop_index != result['loop_num']['UNetModule']:
                print(f'{result["id"]}: drop. loop index = {loop_index}')
                drop_count += 1


class Worker(torch.multiprocessing.Process):
    def __init__(self, input_queue, output_queue, deploy_ready, log_prefix, image_size, device, slo_factor):
        super().__init__()
        self.terminate_receive_flag = False
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.total_request_count = 0

        self.device = device

        self.deploy_ready = deploy_ready
        self.slo_factor = slo_factor

    @torch.no_grad()
    def run(self, **kwargs):
        torch.set_grad_enabled(False)

        # from tp_req_plain_emergency import Model
        # from tp_req_plain_emergency_dynamic_batch_trt import Model
        from plain_continuous_batching import Model
        # from tp_req_plain import Model
        self.model = Model(slo_factor=self.slo_factor)

        self.deploy_ready.release()

        with torch.inference_mode():
            while True:
                while not self.input_queue.empty():
                    request = self.input_queue.get()
                    if request == None:
                        print(
                            f"pid: [{os.getpid()}], received terminate signal!")
                        self.output_queue.put(None)
                        self.terminate_receive_flag = True
                        break
                    self.total_request_count += 1
                    # print(f'request in worker {request}')
                    # request.update({'arrive_time': time.time()})

                    def exc_cb(x, req=request): return (
                        print('ERROR: ', x) or
                        req.update({'exception': x})
                    )

                    def cb(req=request): return (
                        req.update({'cb_time': time.time()}) or
                        self.output_queue.put(req)
                    )
                    self.model(request['id'], request, exc_cb, cb)

                if self.terminate_receive_flag:
                    print(
                        f"pid: [{os.getpid()}], terminate running! total = {self.total_request_count}")
                    break
                time.sleep(0.01)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="parameter for server.")

    parser.add_argument('--image_size', default=256, type=int,
                        help='a value to determine image size')
    parser.add_argument('--rate_scale', required=True,
                        type=str, help='a value to determine arrival rate')
    parser.add_argument('--cv_scale', required=True, type=str,
                        help='a value to determine arrival coefficient of variation')
    parser.add_argument('--slo_scale', required=True, type=str, help='a value to determine slo factor')
    # parser.add_argument('--extra_vae_safety_time', required=True, type=float, help='a value to determine extra budget for vae and safety')
    parser.add_argument('--log_folder', default='log',
                        type=str, help='a value to determine log folder')
    # parser.add_argument('--profile_device', required=True, type=str, help='a value to determine profile device')
    # parser.add_argument('--step_delta', required=True,
    #                     type=float, help='a value to determine running device')

    parser.add_argument('--can_drop', default='1', type=str,
                        help='a value to determine weather to drop requests')
    
    
    args = parser.parse_args()
    
    can_drop = int(args.can_drop)

    # key = f"rate={args.rate_scale}_cv={args.cv_scale}_{args.image_size}"

    # rate = float(args.rate_scale)
    rate = args.rate_scale
    cv = args.cv_scale


    # cv = float(args.cv_scale)

    image_size = args.image_size

    log_folder = args.log_folder

    trace = json.load(open("data/SD_FP16_img256_trace.json"))
    arrival_interval_list = trace[f"rate={float(args.rate_scale):.2f},cv={float(args.cv_scale):.2f}"]
    random_step_list = trace["random_step_list"]
    
    slo_scale = float(args.slo_scale)
    slo_factor = slo_scale

    torch.set_grad_enabled(False)
    try:
        from neustream.test_set import prompt_list

        time_pattern = "request=500"

        from datetime import datetime, timezone, timedelta
        timestamp = datetime.now(
            timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        log_prefix = f"{log_folder}/{timestamp}_image_size={image_size}_{time_pattern}_rate={rate}_cv={cv}"

        input_queue = torch.multiprocessing.Manager().Queue()
        output_queue = torch.multiprocessing.Manager().Queue()

        deploy_ready = torch.multiprocessing.Semaphore(0)

        device = "cuda"
        # 创建工作进程
        worker_list = []
        worker_list.append(Worker(input_queue=input_queue, output_queue=output_queue,  deploy_ready=deploy_ready,
                           log_prefix=log_prefix, image_size=image_size, device=device, slo_factor=slo_factor))

        # trace_time = sum(arrival_interval_list[:workload_request_count])/1.0
        collect_worker = torch.multiprocessing.Process(target=handle_output, args=(
            output_queue, log_prefix, workload_request_count, rate, cv, slo_factor))

        deploy_begin = time.time()
        for _worker in worker_list:
            _worker.start()

        deploy_ready.acquire()
        print(
            f"Workers deploy all done! time used: {time.time() - deploy_begin}")
        emergency_threshold = 1/ref_times[1] * 1

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
            "SLO": slo_scale * (ref_times[0] + ref_times[1] * 50 + ref_times[2]),
            "loop_index": {
                "UNetModule": 0
            },
            "id": -1
        }
        if can_drop:
            warm_up_request_temp['emergency_threshold'] = emergency_threshold

        collect_worker.start()

        # warm up all the unet batch_size
        test_count = 30
        for idx in range(test_count):
            # freeness.acquire()
            warm_up_request = warm_up_request_temp.copy()
            # warm_up_request["id"] = idx
            time.sleep(0.08)
            request_time = time.time()
            warm_up_request["request_time"] = request_time
            warm_up_request['id'] = -1*idx - 1
            if can_drop:
                warm_up_request['iter_deadline'] = request_time + \
                    slo_scale * (ref_times[0] + ref_times[1] * 50)
            input_queue.put(warm_up_request)
        # warm up all the unet batch_size

        # sleep until warm-up end
        time.sleep(20)
        print("warm up succeed!")

        # steps_range = list(range(-5, 6))

        # workload_request_count = 2
        for idx in range(workload_request_count):  # workload_request_count
            # ff = freeness.acquire()
            # print(f"freenes {ff}")
            # continue
            step = random_step_list[idx]
            # slo_factor = slo_factor[idx]
            SLO_THRES = slo_scale * (ref_times[0] +
                                            ref_times[1] * step + ref_times[2])
            time.sleep(arrival_interval_list[idx])
            request_time = time.time()
            
            input = {
                "prompt": prompt_list[idx % 100],
                "height": image_size,
                "width": image_size,
                "loop_num": {
                    "UNetModule": random_step_list[idx],
                },
                "loop_index": {
                    "UNetModule": 0
                },
                "guidance_scale": 7.5,
                "seed": 0,
                "uuid": uuid.uuid1(),
                "request_time": request_time,
                "SLO": SLO_THRES,
                "id": idx,  # for debug
            }
            if can_drop:
                iter_deadline = request_time + slo_scale * \
                    (ref_times[0] + ref_times[1] * random_step_list[idx])
                input['emergency_threshold'] = emergency_threshold
                input['iter_deadline'] = iter_deadline
            
            input_queue.put(input)
            print(
                f"clip_queue put item: {input['id']}\n-----------------------")
            
        time.sleep(10)
        input_queue.put(None)

        for _worker in worker_list:
            _worker.join()

        collect_worker.join()

    except KeyboardInterrupt:
        print("-"*10, "Main process received KeyboardInterrupt", "-"*10)
