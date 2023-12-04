# from __future__ import print_function
import os
import sys
import cv2
import numpy as np
import logging
import threading
from timeit import default_timer as timer
from serve.ttypes import InferenceResult, InferenceStatusEnum ,InferenceParams
from serve import InferenceService
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
from torchpipe.tool import cpp_tools, onnx_tools
import os
import torch
import argparse


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config', dest='config', type=str,  default="./resnet50_gpu_decode.toml",
                    help='configuration file')

args = parser.parse_args()

import torchpipe as tp

def export_onnx(onnx_save_path):
    import torch
    import torchvision.models as models
    resnet50 = models.resnet50().eval()
    x = torch.randn(1,3,224,224)
    onnx_save_path = "./resnet50.onnx"
    tp.utils.models.onnx_export(resnet50, onnx_save_path, x)

class RecognizerHandler():
    def __init__(self, args=None):
        self.label = 'test'

        import time

        onnx_save_path = "./resnet50.onnx"
        if not os.path.exists(onnx_save_path):
            export_onnx(onnx_save_path)

        img_path = "../../../../test/assets/image/gray.jpg"
        img=open(img_path,'rb').read()

        toml_path = args.config 

        from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
        self.nodes = pipe(toml_path)

    
    def forward(self, bin):

        input = {TASK_DATA_KEY: bin.data }

        self.nodes(input)

        max_score, max_class = torch.max(input[TASK_RESULT_KEY], dim=1)
        max_score_float = max_score.item()
        max_class_int = max_class.item()
         
    
        return InferenceResult(status=InferenceStatusEnum.OK, index=max_class_int, score=max_score_float)
        

    def ping(self):
        logging.info('ping()')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path', type=str,default='./resnet50_gpu_decode.toml')
    parser.add_argument('--port', help='train config file path', type=int,default=9002)
    args = parser.parse_args()
    return args


def main(args):
    handler = RecognizerHandler(args)
    processor = InferenceService.Processor(handler)
    transport = TSocket.TServerSocket(port=args.port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
    server.setNumThreads(40)
    print('Starting the server...')
    server.serve()
    print('done.')


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
