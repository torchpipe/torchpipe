
import json
# from helper import exp_config_parser
import omniback
from typing import List
import torch
import torchpipe
import os


target_dir = './25Eurosys-NeuStream-AE/Diffusion/StableDiffusion/RTX4090_SD_FP16_img256/'

# name, cf = exp_config_parser.get_config()
omniback_config = 'data/bs32fake.json'
with open(omniback_config, 'r') as f:
    config = json.load(f)
print(f'model config: {config}')


for m in ['unet', 'vae', 'clip']:
    if not os.path.exists(f'{m}.onnx'):
        model_name = os.path.join(target_dir, f'{m}.onnx')
        import onnxslim
        onnxslim.slim(model_name, f'{m}.onnx', save_as_external_data=True)


# omniback.init('DebugLogger')

def init(model_name=None):
    print(f'init model_name: {model_name}')
    if model_name is None:
        return omniback.pipe(config)
    else:
        sub_config = {model_name:config[model_name]}
        return omniback.pipe(sub_config)