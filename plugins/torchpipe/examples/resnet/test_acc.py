# from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
import requests
import hami
import torchpipe

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = requests.get(url).content







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
       
       
def test_v2():
    import torchpipe.utils.model_helper as helper
    
    model, preprocessor = helper.get_classification_model("resnet50", 224, 224)
    
    onnx_path =  f"{model.__class__.__name__}.onnx"
    if not os.path.exists(onnx_path) and not os.path.exists(onnx_path.replace(".onnx",'.trt')):
        helper.export_x3hw(model, onnx_path, 224, 224)
    
    # onnx_model = OnnxModel(onnx_path, preprocessor)
    config =  {"model":onnx_path, "model::cache":onnx_path.replace(".onnx",'.trt'), "instance_num": '1'}
    config["model"] = onnx_path.replace(".onnx",'.trt')
    hami_model = hami.init("StreamGuard[TensorrtTensor]",config)

    all_result = {}
    dataset = helper.TestImageDataset()
    for data_id, data in dataset:
        if data is not None:
            preprocessed = preprocessor(data).unsqueeze(0)
            with torch.no_grad():
                result = torch.nn.functional.softmax(model(preprocessed), dim=-1)
            # onnx_result = torch.nn.functional.softmax(torch.from_numpy(onnx_model(data)[0]), dim=-1)
            all_result[data_id] = (result, None)
    helper.report_classification(all_result)
    
if __name__ == "__main__":
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





