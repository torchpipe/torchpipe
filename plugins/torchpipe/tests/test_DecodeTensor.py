# filePath: tests/test_DecodeTensor.py
import pytest
import omniback
import torchpipe
import requests
from io import BytesIO
from PIL import Image
import numpy as np
# Function to fetch image from URL
# def fetch_image(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         return response.content
#     except requests.RequestException as e:
#         pytest.skip(f"Skipping test due to failure in fetching image: {e}")

# # Test function
def test_DecodeTensor():
    # Initialize the DecodeTensor model
    model = omniback.init("DecodeTensor")

    img = Image.fromarray(np.zeros((1140, 1200, 3), dtype=np.uint8))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_raw = img_byte_arr.getvalue()

    # Prepare input data
    input = {"data": img_raw}

    # Execute model inference
    model(input)
    
    print(type(input["result"]), input["result"])

    # Assert the shape of the output tensor
    assert input["result"].shape == (1, 3, 1140, 1200)
    
if __name__ =="__main__":
    import time
    # time.sleep(5)
    test_DecodeTensor()