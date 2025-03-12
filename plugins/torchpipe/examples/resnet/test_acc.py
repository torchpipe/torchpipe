# from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
import requests
import hami
import time
# time.sleep(10)
import logging 

# logging.setLevel(logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import torchpipe

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = requests.get(url).content





# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
import torchpipe
import torch
# import cv2
import os
# import torchpipe as tp


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config', dest='config', type=str,  default="./resnet50.toml",
                    help='configuration file')

args = parser.parse_args()


# def export_onnx(onnx_save_path):
#     import torch
#     import torchvision.models as models
#     resnet50 = models.resnet50().eval()
#     x = torch.randn(1,3,224,224)
#     onnx_save_path = "./resnet50.onnx"
#     tp.utils.models.onnx_export(resnet50, onnx_save_path, x)

def get_model(toml_path) :



    interp = hami.init("Interpreter", {"backend": "StreamGuard[DecodeTensor]"})
    return interp

# def test_model():
#     import torchpipe.utils.model_helper as model_helper
    
#     model, preprocessor = model_helper.get_classification_model("resnet50", 224, 224)
#     onnx_path = model_helper.export_x3hw(model, 224, 224)
    
#     onnx_model = OnnxModel(onnx_path)
#     all_result = {}
#     data_id, data = get_data()
#     for item in data:
#         preprocessed = preprocessor(item)
#         result = torch.nn.softmax(model(preprocessed))
    
#         onnx_resust = onnx_model(item)

#         all_result[data_id] = result, onnx_resust
#     report(all_result)
       
import torchpipe

class Torch2Trt:
    def __init__(self, onnx_path, toml_path):
        config = hami.parser.parse(toml_path)
        for k, v in config.items():
            if 'model' in v.keys():
                v['model'] = onnx_path
            v['model::cache'] = onnx_path.replace(".onnx",'.trt')

        dict_config = hami.Dict()
        dict_config['config'] = config
        pipe = hami.create('Interpreter').init({}, dict_config)
        print("config = ",config)
        self.model = pipe
        
    def __call__(self, x):
        data = {'data': x}
        self.model(data)
        return data['result']
    

def test_v2():
    import torchpipe.utils.model_helper as helper
    
    import tempfile
    onnx_path = os.path.join(tempfile.gettempdir(), "resnet50.onnx")
    print(f'testing on {onnx_path}')
    tester = helper.ClassifyModelTester('resnet50', onnx_path)
    
    hami_model  = Torch2Trt(onnx_path, 'config.toml')

    tester.test(hami_model, fix_shape=True)
    
    
if __name__ == "__main__":
    import torchpipe

    import time
    # time.sleep(10)
    test_v2()
    
    raise 1
    import time
    
    # onnx_save_path = "./resnet50.onnx"
    # if not os.path.exists(onnx_save_path):
    #     export_onnx(onnx_save_path)



    toml_path = args.config 
    
    from hami import TASK_DATA_KEY, TASK_RESULT_KEY
    nodes = get_model(toml_path)

    print(type(image))
    
    input = {TASK_DATA_KEY: image}
    nodes(input)

    if TASK_RESULT_KEY not in input.keys():
        print("error : no result")
        
    print(len(input[TASK_RESULT_KEY]))
    print(input[TASK_RESULT_KEY].shape)


 