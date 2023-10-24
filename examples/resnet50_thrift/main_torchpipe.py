# from __future__ import print_function
import numpy as np
from torchvision import models
from timeit import default_timer as timer
from serve.ttypes import InferenceResult
import os
import torchvision.models as models

import torchpipe
import torch
try:
    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
except Exception as e:
    print("import torchpipe failed: ", e)


class RecognizerHandler():
    def __init__(self, args=None):
        self.cls_trt_max_batchsize = 16
        self.resize_size = 320 
        self.fp16 = True
        resnet50 = resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
        x = torch.randn(1,3,224,224)
        onnx_save_path = "./resnet50.onnx"
        if not os.path.exists(onnx_save_path):
            torch.onnx.export(resnet50,
                            x,
                            onnx_save_path,
                            opset_version=17,
                            do_constant_folding=True,
                            input_names=["input"],
                            output_names=["output"], 
                            dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}})
        

        config = torchpipe.parse_toml("resnet50.toml")
        self.classification_engine = pipe(config)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        print("finish init")

    def infer_batch(self, bin_list):
        results = InferenceResult()
        bin_data = {TASK_DATA_KEY:bin_list.data, "node_name":"cpu_decoder"}
        try:
            self.classification_engine(bin_data)
            if TASK_RESULT_KEY not in bin_data.keys():
                print("error decode")
                return results
            else:
                dis = self.softmax(bin_data[TASK_RESULT_KEY])
            
            maxprob,label_index = torch.max(dis,1)

            label_index_tmp = label_index.item()
            maxprob_tmp = maxprob.item()
            
            results.label = str(label_index_tmp)
            results.score = float(maxprob_tmp)
            return results


        except Exception as e:
            print("uuid={:} : {}".format(bin_list.uuid, e))
            return results
    


    def ping(self):
        print('ping()')

