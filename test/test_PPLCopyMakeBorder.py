import torchpipe


import cv2
import torch
import pytest


@pytest.mark.skipif(not torchpipe.libipipe.is_registered('PPLCopyMakeBorderTensor'), reason="PPLCopyMakeBorderTensor backend is not registered")
def test_PPLCopyMakeBorderTensor():
    import numpy as np
    img_path = "assets/image/gray.jpg"
    img_raw = open(img_path, 'rb').read()
    img = cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
    print(img.shape)  # (274, 442, 3)
    resize_h = 480
    resize_w = 222

    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
    nodes = torchpipe.pipe({
        "backend": "S[DecodeTensor,PPLResizeTensor ,PPLCopyMakeBorderTensor,SyncTensor]",
        "resize_h": resize_h,
        "resize_w": resize_w,
        "data_format": "hwc"})

    import numpy as np

    cv2_result = cv2.resize(img, (resize_w, resize_h))
    cv2_result = cv2.copyMakeBorder(
        cv2_result, 5, 15, 25, 35, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    input = {TASK_DATA_KEY: img_raw,
             "top": 5,
             "bottom": 15,
             "left": 25,
             "right": 35}
    nodes(input)

    # img = input["result"].cpu().numpy()
    img = input["result"].cpu().numpy()
    print(img.shape, cv2_result.shape)
    assert (img.shape == cv2_result.shape)

    # cv2.imwrite("debug_cv2.jpg",cv2_result)
    # cv2.imwrite("debug.jpg",img)
    diff = cv2.absdiff(img, cv2_result)
    non_zero_diff = diff[np.nonzero(diff)]
    mean_diff = np.mean(np.abs(non_zero_diff))
    print(mean_diff)
    assert (mean_diff < 1.2)


if __name__ == "__main__":
    import time
    # time.sleep(8)

    test_PPLCopyMakeBorderTensor()
