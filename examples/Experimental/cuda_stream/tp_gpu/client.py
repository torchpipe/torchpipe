import sys
import torch
import cv2
import os
import numpy as np

from serve.ttypes import InferenceResult ,InferenceParams

MODEL_NAME="resnet50_thrift"

class ThriftInfer:
    """wrapper for thrift's python API. You may need to re-implement this class."""

    def __init__(self, host="localhost", port=9002) -> None:
        """

        :param host: ip
        :type host: str
        :param port: port
        :type port: int
        """
        from serve import InferenceService
        from serve.ttypes import InferenceParams

        self.InferenceParams = InferenceParams

        from thrift.transport import TSocket
        from thrift.transport import TTransport
        from thrift.protocol import TBinaryProtocol

        self.transport = TSocket.TSocket(host, port)
        self.transport = TTransport.TBufferedTransport(self.transport)
        self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)

        self.client = InferenceService.Client(self.protocol)

        # Connect!
        self.transport.open()
        self.client.ping()

    def forward(self, data):
        """batch processing

        :param data: batched data
        :type data: (str, bytes)
        :return:
        :rtype: Any
        """
        img_path, img_bytes = data[0]
        result: InferenceResult= self.client.forward(self.InferenceParams(uuid="1", data=img_bytes))
        max_value, max_index = result.score, result.index
        return max_value, max_index


    def __del__(self):
        self.transport.close()

 


if __name__ == "__main__":

    img_path = "../../../../test/assets/image/gray.jpg"
    img=open(img_path,'rb').read()

    num_clients = 40
    forwards = [ThriftInfer() for i in range(num_clients)]
    forwards[0].forward([(img_path, img)])

    from torchpipe.utils.test import test_from_raw_file
    result = test_from_raw_file([x.forward for x in forwards], os.path.join("../../../..", "test/assets/encode_jpeg/"),num_clients=num_clients, batch_size=1,total_number=40000)


    print("\n", result)

    if True:
        import pickle
        pkl_path = MODEL_NAME+".pkl"# toml_path.replace(".toml",".pkl")
        with open(pkl_path,"wb") as f:
            pickle.dump(result, f)
        print("save result to ", pkl_path)
