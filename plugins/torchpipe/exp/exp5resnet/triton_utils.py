# import sys
# import torch
import cv2
import os
import numpy as np

from tritonclient.grpc import InferenceServerClient, InferRequestedOutput, InferInput


# MODEL_NAME = "ensemble_dali_resnet"
RETURN_A = os.environ.get("RETURN_A", "0") == "1"
RETURN_B = os.environ.get("RETURN_B", "0") == "1"
RETURN_C = os.environ.get("RETURN_C", "0") == "1"
RETURN_D = os.environ.get("RETURN_D", "0") == "1"


class TritonInfer:
    def __init__(self,model_name):

        # import torchpipe as tp

        self.input_name = "INPUT"
        self.output_name = "OUTPUT"

        self.model_name = model_name# "ensemble_dali_resnet"
        print("self.model_name  ", self.model_name)
        self.url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
        self.triton_client = InferenceServerClient(url=self.url)
        self.outputs = [InferRequestedOutput(self.output_name)]

    def forward(self, img):
        img_path, img_bytes = img[0]
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)[None, :]
        # print(len(img_np))
        try:
            inputs = [InferInput(self.input_name, img_np.shape, "UINT8")]

            inputs[0].set_data_from_numpy(img_np)

            triton_results = self.triton_client.infer(
                model_name=self.model_name, inputs=inputs, outputs=self.outputs
            )
        except Exception as e:
            print(e)
        arr = triton_results.as_numpy(self.output_name)
        return 0
        max_value = np.max(arr)
        max_index = np.argmax(arr)
        # print(max_index, max_value)
        return max_value, max_index



class ProcessAdaptor:
    def __init__(self, class_def, args):
        from multiprocessing import Process, Queue, Event

        self.class_def = class_def
        self.args = args

        self.queue = Queue()
        self.event = Event()
        self.instance = Process(target=self.run)
        self.alive = Event()

        self.instance.start()

    def forward(self, data):
        self.queue.put(data)
        # while not self.queue.empty():
        self.event.wait()
        self.event.clear()

    def run(self):
        self.target = self.class_def(self.args)
        while not self.alive.is_set():
            try:
                p = self.queue.get(block=True, timeout=2)
                if p is None:
                    continue
            except:
                continue
            self.target.forward(p)
            self.event.set()
        # print(f'ProcessAdaptor for {self.class_def} stopped.')

    def close(self):
        self.alive.set()
        self.queue.put(None)
        # print(f'ProcessAdaptor for {self.class_def} closing...')
        self.instance.join()

    def __del__(self):
        self.close()
        
    @staticmethod
    def close_all(clients):

        from concurrent.futures import ThreadPoolExecutor

        def close_client(client):
            if hasattr(client, "close"):
                client.close()
                # print("client closed")

        # Assuming clients is a list of your client objects
        with ThreadPoolExecutor() as executor:
            executor.map(close_client, clients)


class TritonWithPreprocess:
    def __init__(self, model_name):

        self.input_name = "input"
        self.output_name = "output"

        self.model_name = model_name
        self.url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
        self.triton_client = InferenceServerClient(url=self.url)
        self.outputs = [InferRequestedOutput(self.output_name)]
        self.mean = np.array((0.485, 0.456, 0.406)).astype(np.float32)
        self.std = np.array((0.229, 0.224, 0.225)).astype(np.float32)

    def forward(self, img):
        img_path, img_bytes = img[0]

        img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))

        img_np = (
                (img[:, :, ::-1].astype(np.float32) - self.mean) / self.std
            ).transpose((2, 0, 1))[None, ...]

        inputs = [InferInput(self.input_name, img_np.shape, "FP32")]

        if RETURN_A:
            return 0, 0
        inputs[0].set_data_from_numpy(img_np)

        if RETURN_B:
            return 0, 0

        triton_results = self.triton_client.infer(
            model_name=self.model_name, inputs=inputs, outputs=self.outputs
        )
        if RETURN_C:
            return 0, 0

        arr = triton_results.as_numpy(self.output_name)

        if RETURN_D:
            return 0, 0
        # print(arr.shape)
        return 0
        max_value = np.max(arr)
        max_index = np.argmax(arr)
        # print(max_index, max_value)
        return max_value, max_index


def get_clients(model_name, num_clients):
    return [TritonInfer(model_name) for x in range(num_clients)]


def get_clients_with_preprocess(model_name, num_clients, use_process):
    if not use_process:
        return [TritonWithPreprocess(model_name) for x in range(num_clients)]
    else:
        return [
            ProcessAdaptor(TritonWithPreprocess, model_name)
            for x in range(num_clients)
        ]
