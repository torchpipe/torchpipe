# filePath: tests/test_decodeMat.py
import pytest
import omniback
import torchpipe
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
# Function to fetch image from URL
# def fetch_image(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         return response.content
#     except requests.RequestException as e:
#         pytest.skip(f"Skipping test due to failure in fetching image: {e}")

# # Test function


def test_decodeMat():
    # Initialize the DecodeTensor model
    model = omniback.init("S[DecodeMat,Mat2Tensor]", {"color": "rgb"})

    img = np.ones((1, 1, 3), dtype=np.uint8)*5
    img[0, 0, 0] = 0
    img[0, 0, 1] = 0
    img[0, 0, 2] = 255
    img = Image.fromarray(img)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_raw = img_byte_arr.getvalue()

    # Prepare input data
    input = {"data": img_raw}

    # Execute model inference
    model(input)
    result = input["result"]
    print(type(result.shape))
    if not isinstance(result, torch.Tensor):
        result = torch.from_dlpack(result)
    # Assert the shape of the output tensor
    assert result.shape == (1, 1, 3)
    # print(input["result"].shape)
    # print(input["result"][0,0,:])
    assert result[0, 0, -1] == 0


if __name__ == "__main__":
    import time
    # time.sleep(5)
    test_decodeMat()
