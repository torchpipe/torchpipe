import sys
import torch
import cv2
import os
import numpy as np

import tritongrpcclient

class TritonInfer:
    def __init__(self, args):
        
        import torchpipe as tp
        
        
        self.input_name = "INPUT"
        self.output_name = "OUTPUT"
 
        self.model_name = "ensemble_dali_resnet"
        self.url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
        self.triton_client = tritongrpcclient.InferenceServerClient(url=self.url)
        self.outputs = tritongrpcclient.InferRequestedOutput(self.output_name)

        # # add

        # # import pdb; pdb.set_trace()

        # if args.input_data_type == 'string':
        #     in_0 = np.array(list("helloworld"), dtype=np.object_)

        # elif args.input_data_type == 'fp32_chw':
        #     in_0 = get_img_fp32_chw()
        
        # elif args.input_data_type == 'uint8':
        #     in_0 = get_img_uint8()

        # elif args.input_data_type == 'string_uint8':
        #     in_0 = get_string_uint8()
            
        # else:
        #     print('error : load data type error')
        #     self.in_0 = None

        # if args.batch == 'True' or args.batch == 'true':
        #     self.input_data = in_0[np.newaxis, :]
        # else:
        #     self.input_data = in_0
        
        # if args.input_data_type == 'fp32_chw':
        #     self.inputs = grpcclient.InferInput(self.input_name, self.input_data.shape , datatype="FP32")
        #     self.inputs.set_data_from_numpy(self.input_data.astype(np.float32))
        # elif args.input_data_type == 'uint8' or args.input_data_type == 'string_uint8':
        #     self.inputs = grpcclient.InferInput(self.input_name, self.input_data.shape , datatype="UINT8")
        #     self.inputs.set_data_from_numpy(self.input_data.astype(np.uint8))
        # elif args.input_data_type == 'string':
        #     self.inputs = grpcclient.InferInput(self.input_name, self.input_data.shape , datatype="BYTES")
        #     self.inputs.set_data_from_numpy(self.input_data.astype(np.object_))
        # else:
        #     print('error: input data type set in cfg error! please check')
        #     exit(0)



    def run(self, img):
        img_path, img_bytes = img[0]

        
        inputs = [tritongrpcclient.InferInput(self.input_name, [len(img_bytes)], "UINT8")]
        # outputs.append(tritongrpcclient.InferRequestedOutput(self.output_name))

        inputs[0].set_data_from_numpy(np.array(img_bytes, dtype=np.uint8))
        
        triton_results = self.triton_client.infer(
            model_name=self.model_name,
            inputs=[inputs],
            outputs=[self.outputs])


        output = triton_results.as_numpy( self.output_name )
        print(output)
        return output