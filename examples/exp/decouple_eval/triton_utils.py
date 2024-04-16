import sys
import torch
import cv2
import os
import numpy as np

from tritonclient.grpc import InferenceServerClient, InferRequestedOutput, InferInput


MODEL_NAME = "ensemble_dali_resnet"


class TritonInfer:
    def __init__(self):

        import torchpipe as tp

        self.input_name = "INPUT"
        self.output_name = "OUTPUT"

        self.model_name = "ensemble_dali_resnet"
        self.url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
        self.triton_client = InferenceServerClient(url=self.url)
        self.outputs = [InferRequestedOutput(self.output_name)]

    def forward(self, img):
        img_path, img_bytes = img[0]
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)[None, :]
        # print(len(img_np))

        inputs = [InferInput(self.input_name, img_np.shape, "UINT8")]
        # outputs.append(tritongrpcclient.InferRequestedOutput(self.output_name))

        inputs[0].set_data_from_numpy(img_np)

        triton_results = self.triton_client.infer(
            model_name=self.model_name, inputs=inputs, outputs=self.outputs
        )

        arr = triton_results.as_numpy(self.output_name)
        # print(arr.shape)
        max_value = np.max(arr)
        max_index = np.argmax(arr)
        # print(max_index, max_value)
        return max_value, max_index


def get_clients(model_name, num_clients):
    return [TritonInfer() for x in range(num_clients)]
