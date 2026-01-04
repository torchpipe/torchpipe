# from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
import requests
import omniback
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



    interp = omniback.init("Interpreter", {"backend": "StreamGuard[DecodeTensor]"})
    return interp

       
import torchpipe

class Torch2Trt:
    def __init__(self, onnx_path, toml_path):
        config = omniback.parser.parse(toml_path)
        for k, v in config.items():
            if 'model' in v.keys():
                v['model'] = onnx_path
            v['model::cache'] = onnx_path.replace(".onnx",'.trt')

        kwargs = omniback.Dict()
        kwargs['config'] = config
        pipe = omniback.create('Interpreter').init({}, kwargs)
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
    
    ms_val_dataset = helper.get_mini_imagenet()
    # next(iter(ms_val_dataset))
    
    true_labels = []
    pred_labels = []
    # print(ms_val_dataset.class_to_index)
    # import pdb; pdb.set_trace()
    helper.import_or_install_package('tqdm')
    from tqdm import tqdm

    tester.model.cuda()
    for item in tqdm(ms_val_dataset, desc="Processing", position=0, leave=True):
        image_file = item['image:FILE']
        category = item['category']
        infer_cls, infer_score = tester(image_file)

        true_labels.append(category)
        pred_labels.append(infer_cls)
    # import pdb; pdb.set_trace()
    map_label = helper.align_labels(true_labels, pred_labels, 100)

    metric = helper.evaluate_classification(true_labels, [map_label[x] for x in pred_labels])
    assert metric.get('f1', 0) >= 0.89
    # omniback_model  = Torch2Trt(onnx_path, 'config.toml')

    # - Accuracy:  0.8713
    # - Precision: 0.9386
    # - Recall:    0.8627
    # - F1 Score:  0.8975
    
    
if __name__ == "__main__":
    import torchpipe

    import time
    # time.sleep(10)
    test_v2()
    
    
 