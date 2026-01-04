import os
import tempfile
import logging
import time
import pytest

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Check required dependencies
cv2_numpy_available = True
try:
    import cv2
    import numpy as np
except ImportError:
    cv2_numpy_available = False

# Other imports
import torch
import omniback
import torchpipe
import torchpipe.utils.model_helper as helper
from PIL import Image


class Torch2Trt:
    def __init__(self, onnx_path, toml_path):
        config = omniback.parser.parse(toml_path)
        for k, v in config.items():
            if 'model' in v.keys():
                v['model'] = onnx_path
            v['model::cache'] = onnx_path.replace(".onnx", '.trt')+".encrypt"

        kwargs = omniback.Dict()
        kwargs['config'] = config
        pipe = omniback.create('Interpreter').init({}, kwargs)
        logger.info(f"config = {config}")
        self.model = pipe
        
    def __call__(self, x):
        data = {'data': x}
        self.model(data)
        re = data['result']
        if not isinstance(re, torch.Tensor):
            re = torch.from_dlpack(re)
        return re


def get_model(toml_path):
    interp = omniback.init("Interpreter", {"backend": "StreamGuard[DecodeTensor]"})
    return interp


@pytest.mark.skipif(not cv2_numpy_available, 
                   reason="Test requires cv2 and numpy libraries")
def test_resnet50_classification():
    """Test ResNet50 image classification"""
    # Prepare test model path
    onnx_path = os.path.join(tempfile.gettempdir(), "pytest_resnet50.onnx")
    logger.info(f'Testing with model: {onnx_path}')
    
    # Initialize test utilities and model
    tester = helper.ClassifyModelTester('resnet50', onnx_path)
    omniback_model = Torch2Trt(onnx_path, 'config/resnet50.toml')
    
    # Run test
    tester.test(omniback_model)
    # try:
    #     import shutil
    #     shutil.rmtree(os.path.dirname(onnx_path))
    # except:
    #     pass


if __name__ == "__main__":
    pytest.main(['-xvs', __file__])