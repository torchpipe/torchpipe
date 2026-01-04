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


# import torch

# import cv2
import os

# import torchpipe as tp
try:
    import omniback
except:
    pass

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    dest="model",
    type=str,
    default="resnet18",
    help="model name",
)
parser.add_argument(
    "--client", dest="client", type=int, default=40, help="number of clients"
)
parser.add_argument(
    "--save", dest="save", type=str, default="", help="save result to pickle"
)
parser.add_argument('--debug', action='store_true', help='enable debug mode')

parser.add_argument(
    "--trt_instance_num",
    dest="trt_instance_num",
    type=int,
    default=5,
    help="number of instances",
)
parser.add_argument(
    "--max",
    dest="max",
    type=int,
    default=8,
    help="max batch size",
)
parser.add_argument(
    "--timeout",
    dest="timeout",
    type=int,
    default=5,
    help="max batch size",
)
parser.add_argument(
    "--total_number",
    dest="total_number",
    type=int,
    default=0,
    help="number of clients",
)
parser.add_argument(
    "--preprocess",
    dest="preprocess",
    type=str,
    default="cpu",
    help="cpu preprocess or gpu preprocess",
)
parser.add_argument(
    "--preprocess-instances",
    dest="preprocess_instances",
    type=int,
    default=0,
    help="cpu preprocess or gpu preprocess instance",
)
parser.add_argument(
    "--prob",
    dest="prob",
    type=float,
    default=0.5,
    help="probability for cpu preprocess in mix mode",
)

args = parser.parse_args()


def export_onnx(onnx_save_path, model_name):
    import torch
    import torchvision.models as models
    import timm
    import torchpipe as tp

    # resnet101 = tp.utils.model_helper.create_model(name).eval()
    if model_name in timm.list_models():
        model = timm.create_model(
            model_name, pretrained=False, exportable=True).eval()
    elif "faster_vit" in model_name:
        assert timm.__version__ == "0.9.6"
        import fastervit

        # timm==0.9.6

        model = fastervit.create_model(model_name, pretrained=False).eval()
    else:
        print(f"skip export {model_name}")
        return
    x = torch.randn(1, 3, 224, 224)
    onnx_save_path = f"./{model_name}.onnx"
    tp.utils.model_helper.export_n3hw(model, onnx_save_path, 224, 224)


def get_cpu_preprocess_cfg(preprocess_instances):
    if preprocess_instances == 0:
        preprocess_instances = 24
    return {
        "backend": "S[DecodeMat,ResizeMat,CvtColorMat,Mat2Tensor,StreamGuard]",
        "resize_h": "224",
        "resize_w": "224",
        "instance_num": preprocess_instances,
        "color": "rgb",
        'batching_timeout': "0",
    }


def get_gpu_preprocess_cfg(preprocess_instances):
    if preprocess_instances == 0:
        preprocess_instances = 11
    return {
        "backend": "S[DecodeTensor,ResizeTensor,CvtColorTensor,StreamGuard]",
        "resize_h": "224",
        "resize_w": "224",
        "instance_num": preprocess_instances,
        "color": "rgb",
    }


def get_mix_preprocess_cfg(probability=0.5):
    return {'node_entrypoint': f'Register[Random({probability})[node.cpu,node.gpu]]'
            }


def get_config(args):
    model_name = args.model
    if args.preprocess == "gpu":
        preprocess = get_gpu_preprocess_cfg(args.preprocess_instances)
    elif args.preprocess == "mix":
        preprocess = get_mix_preprocess_cfg(args.prob)
    else:
        preprocess = get_cpu_preprocess_cfg(args.preprocess_instances)

    preprocess["next"] = model_name
    config = {
        "jpg_decoder": preprocess,
        model_name: {
            "backend": "S[TensorrtTensor, StreamGuard]",
            "instance_num": args.trt_instance_num,
            "max": args.max,
            "mean": "123.675, 116.28, 103.53",
            "model": f"./{model_name}.onnx",
            "std": "58.395, 57.120, 57.375",
            "model::cache": f"./{model_name}_b{args.max}i{args.trt_instance_num}.trt",
            "post_processor": "CpuTensor",
        },
        "global": {"batching_timeout": args.timeout},
    }
    if args.preprocess == "mix":
        config['gpu'] = get_gpu_preprocess_cfg(args.preprocess_instances)
        config['cpu'] = get_cpu_preprocess_cfg(args.preprocess_instances)

    if model_name == "empty":
        del config[model_name]
        del config["jpg_decoder"]["next"]

    return config


if __name__ == "__main__":

    import time

    # time.sleep(5)
    
    if args.debug:
        omniback.init("DebugLogger")
    config = {}
    onnx_save_path = f"./{args.model}.onnx"
    if (
        not os.path.exists(onnx_save_path)
        and args.model != "empty"
        and "triton" not in args.model
        and "ensemble" not in args.model
    ):
        export_onnx(onnx_save_path, args.model)

    def run(img):
        for img_path, img_bytes in img:
            input = {"data": img_bytes, "node_name": "jpg_decoder"}
            nodes(input)

            if "result" not in input.keys():
                print("error : no result")
                return
            # z = input["result"].cpu()

    def only_preprocess(img):
        for img_path, img_bytes in img:
            input = {"data": img_bytes, "node_name": "jpg_decoder"}
            nodes(input)

            if "result" not in input.keys():
                print("error : no result")
                return

    clients = []
    if "ensemble" in args.model or args.model == "triton_resnet_ensemble":
        import triton_utils

        clients = triton_utils.get_clients(args.model, args.client)
        run = [x.forward for x in clients]
    elif args.model == "triton_resnet":
        import triton_utils

        clients = triton_utils.get_clients_with_preprocess(
            "resnet_trt", args.client)
        run = [x.forward for x in clients]
    else:
        config = get_config(args)
        print(config)
        import torchpipe as tp

        nodes = omniback.pipe(config)

    if args.model == "empty":
        print("args.model is empty. test preprocess only")
        run = only_preprocess

    # run([(img_path, img)])

    from test_tools import test_from_raw_file

    total_number = args.total_number
    if total_number == 0:
        if args.client == 1:
            total_number = 5000
        else:
            total_number = 10000

    img_dir = 'encode_jpeg'
    # img_dir = '320'
    result = test_from_raw_file(
        run,
        os.path.join("../", "tests/assets/", img_dir),
        num_clients=args.client,
        total_number=total_number,
    )

    keep = {
        "throughput::qps": "QPS",
        "latency::TP99": "TP99",
        "latency::TP50": "TP50",
        "latency::TP99.9": "TP99.9",
        "latency::TP99.99": "TP99.99",
        "latency::TP50": "TP50",
        "gpu_usage": "GPU Usage",
        "cpu_usage": "CPU Usage",
    }

    print("\n", result)

    new_result = {keep[k]: v for k, v in result.items() if k in keep.keys()}
    print("\n\n")
    print({args.client: new_result})
    print("\n\n")

    from test_tools import ProcessAdaptor

    ProcessAdaptor.close_all(clients)

    if args.save:
        if args.save.endswith('.txt') or args.save.endswith('.log'):
            result_str = f'model:{args.model},'
            result_str += f'preprocessor:{args.preprocess},'
            for k, v in result.items():
                result_str += f'{k}:{v},'
            result_str = result_str.strip(',')
            result_str +='\n'
            with open(args.save, "a") as f:
                f.write(result_str)
        else:
            import pickle

            with open(args.save, "wb") as f:
                pickle.dump({args.client: result}, f)
