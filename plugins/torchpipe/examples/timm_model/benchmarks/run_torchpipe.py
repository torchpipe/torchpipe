
import fire
import time
from torchpipe import pipe
import torchpipe
import timm
import subprocess

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
 
def get_client(image_path, model_path):
    data = open(image_path, 'rb').read()
    
    thread_safe_pipe = pipe({
        "preprocessor": {
            "backend": "S[DecodeTensor,ResizeTensor,CvtColorTensor,SyncTensor]", # 2 gpu instance are enough
            # "backend": "S[DecodeMat,ResizeMat,CvtColorMat,Mat2Tensor,SyncTensor]",
            'instance_num': 2,
            'color': 'rgb',
            'resize_h': '448',
            'resize_w': '448',
            'next': 'model',

        },
        "model": {
            "backend": "SyncTensor[TensorrtTensor]",
            "model": model_path,
            "model::cache": model_path.replace(".onnx", ".trt"),
            "max": '4',
            'batching_timeout': 4,  # ms, timeout for batching
            'instance_num': 2,
            'mean': "123.675, 116.28, 103.53",
            'std': "58.395, 57.120, 57.375",  # merged into trt
        }}
    )

    def forward_func(ids):
        ldata = {'data': data}
        thread_safe_pipe(ldata)
        return ldata['result']

    return forward_func

class ThriftClients:
    def __init__(self, model_path, port, num_clients=1):
        from thrift import Thrift
        from thrift.transport import TSocket
        from thrift.transport import TTransport
        from thrift.protocol import TBinaryProtocol
        
        self.server = subprocess.Popen(
            ["python", "-u", "benchmarks/server_torchpipe.py", 
             "start_thrift_server", "--model_path", model_path, "--port", str(port)],
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
        )

        from image_processing import ImageProcessingService
        self.ImageProcessingService = ImageProcessingService

        # Make socket
        self.clients = []
        while True:
            try:
                soc = TSocket.TSocket('localhost', port)
                soc.setTimeout(1000)
                self.transport = TTransport.TBufferedTransport(soc)
                protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
                self.client = ImageProcessingService.Client(protocol)
                self.transport.open()
                self.transport.close()
                del self.client
                del self.transport
                break
            except Exception as e:
                print(f"Client: Waiting for thrift server to start. port={port}. Sleep and wait.")
                time.sleep(8)
                continue
        for i in range(num_clients):
            soc = TSocket.TSocket('localhost', port)
            transport = TTransport.TBufferedTransport(soc)
            protocol = TBinaryProtocol.TBinaryProtocol(transport)
            client = ImageProcessingService.Client(protocol)
            transport.open()
            self.clients.append((client, transport, soc))
            print(f'connect the {i}th client')

    def forward(self, data, i):
        request = self.clients[i][0].process_image(
            self.ImageProcessingService.ProcessRequest(data))
        if not request.success:
            raise RuntimeError(f"Thrift request failed: {request.error_message}")
        return request.result_data
    
    def __del__(self):
        self.server.terminate()
        
        
def get_thrift_client(image_path, model_path, num_clients=1, port=9090):
    data = open(image_path, 'rb').read()
    client = ThriftClients(model_path, port=port, num_clients=num_clients)

    forward_funcs = []
    for i in range(num_clients):
        def make_func(i):
            def func(ids):
                # print(i, ids)
                return client.forward(data, i)
            return func
        forward_funcs.append(make_func(i))
    return forward_funcs, client

def main(
    model_name: str = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    image_path: str = "../../tests/assets/encode_jpeg/grace_hopper_517x606.jpg",
    precision: str = "fp16",  # "fp32" or "fp16"
    max_batch_size: int = 1,
    use_cache: bool = True,
    cache_path: str = "./.cache/torchpipe",
):
    """
    Convert and run inference on timm model using torch2trt.
    
    Args:
        model_name: Name of the timm model to load
        image_path: Path to input image
        batch_size: Batch size for inference
        precision: TensorRT precision ("fp32" or "fp16")
        max_batch_size: Maximum batch size for TRT engine
        use_cache: Whether to cache the TRT engine
        cache_path: Directory to store cached engines
        benchmark: Whether to run benchmarking
        concurrent: Whether to run concurrent inference
        num_concurrent: Number of concurrent inference requests
        num_benchmark_runs: Number of runs for benchmarking
    """
    pass


if __name__ == "__main__":
    fire.Fire(main)
