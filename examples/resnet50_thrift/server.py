from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from serve import InferenceService


logging.basicConfig(level=logging.DEBUG)
CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='Port to listen.', default=8888)
    parser.add_argument('--torchpipe', type=int, help='Port to listen.', default=0)
    args = parser.parse_args(argv)
    return args


def main(args):
    if args.torchpipe == 1:
        from main_torchpipe import RecognizerHandler
    else:
        from main_trt import RecognizerHandler

    handler = RecognizerHandler(args)
    processor = InferenceService.Processor(handler)
    transport = TSocket.TServerSocket(port=args.port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
    print('Starting the server...')

    server.setNumThreads(20)

    server.serve()
    print('done.')


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))