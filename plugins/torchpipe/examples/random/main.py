from typing import List
import omniback
import torchpipe


model = omniback.pipe('random.toml')

import cv2
img_path = '../../tests/assets/640/640.jpg'
img = cv2.imread(img_path)
assert img is not None, f"Image not found at {img_path}"
with open(img_path, 'rb') as f:
    data = f.read()

CONFIG = {}
def get_test_function():
    
    io = {"data": data, 'node_name':'mix'}
    model(io)
    
    def forward(input : List[int]):
        io =  {"data": data, 'node_name': CONFIG['node']}
        model(io)
        result = io[("result")]
        # print(f"Processed image shape: {result.shape}")
        return 0
    return forward


def main(node = "mix"):
    CONFIG['node'] = node
    # omniback.init("DebugLogger")
    # import time
    # time.sleep(5)
    omniback.utils.test.test_from_ids(forward_function=[get_test_function() for _ in range(10)],
                                ids=[0]*1000,
                                request_batch = 1)
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)