import sys
import torch
import cv2
import os
import numpy as np

import tritonclient.grpc as grpcclient
MODEL_NAME="ensemble_model"
class TritonInfer:
    def __init__(self):
        
        import torchpipe as tp
        
        
        self.input_name = "input_image"
        self.output_name = "recognized_text"
 
        self.model_name = MODEL_NAME
        self.url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
        self.triton_client = grpcclient.InferenceServerClient(url=self.url)
        self.outputs = [grpcclient.InferRequestedOutput(self.output_name)]


    def forward(self, img):
        img_path, img_bytes = img[0]
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)[None, :]
        #print(len(img_np))

        
        inputs = [grpcclient.InferInput(self.input_name, img_np.shape, "UINT8")]
        # outputs.append(grpcclient.InferRequestedOutput(self.output_name))

        inputs[0].set_data_from_numpy(img_np)
        
        triton_results = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=self.outputs)

        arr = triton_results.as_numpy( self.output_name )
        #print(arr.shape)
        max_value = np.max(arr)
        max_index = np.argmax(arr)
        #print(max_index, max_value)
        return max_value, max_index


if __name__ == "__main__":

    img_path = "img1.jpg"
    img=open(img_path,'rb').read()

    num_clients = 40
    forwards = [TritonInfer() for i in range(num_clients)]
    forwards[0].forward([(img_path, img)])

    from torchpipe.utils.test import test_from_raw_file
    result = test_from_raw_file([x.forward for x in forwards], os.path.join("./", "."),num_clients=40, batch_size=1,total_number=40000)


    print("\n", result)

    if True:
        import pickle
        pkl_path = MODEL_NAME+".pkl"# toml_path.replace(".toml",".pkl")
        with open(pkl_path,"wb") as f:
            pickle.dump(result, f)
        print("save result to ", pkl_path)



# client = grpcclient.InferenceServerClient(url="localhost:8001")

# image_data = np.fromfile("img1.jpg", dtype="uint8")
# image_data = np.expand_dims(image_data, axis=0)

# input_tensors = [grpcclient.InferInput("input_image", image_data.shape, "UINT8")]
# input_tensors[0].set_data_from_numpy(image_data)
# results = client.infer(model_name="ensemble_model", inputs=input_tensors)
# output_data = results.as_numpy("recognized_text").astype(str)
# print(output_data)
