from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import sys
import time
import os
import math
import numpy as np
import argparse
import cv2
from serve import InferenceService
from serve.ttypes import InferenceParams

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from timeit import default_timer as timer

CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def main(args):
    # Make socket
    transport = TSocket.TSocket(args.host, args.port)

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)

    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    client = InferenceService.Client(protocol)

    # Connect!
    transport.open()
    client.ping()

    list_images = ["./img/dog.jpg"]
    print(len(list_images))
    data_num=0

    all_time = 0 

    for img_path in list_images:
        try :
            # import pdb;pdb.set_trace()
            img = cv2.imread(img_path, 1)
            img = cv2.imencode('.jpg',img)[1]
            stat_time = timer()
            results = client.infer_batch(InferenceParams(img_path, img.tobytes()))
            print("results: ", results)
            all_time += timer()-stat_time

        except Exception as e:
            print(e)

    print("="*50)
    print("Im ok ~~~~ me me da ~~~~",all_time)
    print(data_num)
    print("="*50)
    # Close!
    transport.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='Port to listen.', default=8888)
    parser.add_argument('--host',type=str,help='Host to run service',default='localhost')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

