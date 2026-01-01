# filePath: tests/test_DecodeTensor.py
import pytest
import omniback
import torchpipe
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch

def test_DecodeTensor():
    # Initialize the DecodeTensor model
    model = omniback.init("S_v0[DecodeMat,CvtColorMat,ResizeMat,Mat2Tensor]", {"color": "bgr", "resize_h":"221","resize_w":'110'})

    img = np.ones((1, 1, 3), dtype=np.uint8)*5
    img[0,0,0] = 0
    img[0,0,1] = 0
    img[0,0,2] = 255
    img = Image.fromarray(img)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_raw = img_byte_arr.getvalue()

    # Prepare input data
    input = {"data": img_raw}

    # Execute model inference
    model(input)

    result = input["result"]
    if not isinstance(result, torch.Tensor):
        result = torch.from_dlpack(result)
    # Assert the shape of the output tensor
    assert result.shape == (221, 110, 3)
    # print(input["result"])
    # print(input["result"][0,0,:])
    assert result[0, 0, -1] == 0
    
if __name__ == "__main__":
    test_DecodeTensor()