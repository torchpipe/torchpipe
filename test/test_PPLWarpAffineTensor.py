import torchpipe


import cv2
import torch
import pytest

# 在torchpipe.libipipe中没有注册PPLWarpAffineTensor后端时跳过pytest测试


@pytest.mark.skipif(not torchpipe.libipipe.is_registered('PPLWarpAffineTensor'), reason="PPLWarpAffineTensor backend is not registered")
def test_PPLWarpAffineTensor():
    import numpy as np
    img_path = "assets/image/gray.jpg"
    img_raw = open(img_path, 'rb').read()
    img = cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
    print(img.shape)  # (274, 442, 3)
    target_h = 224
    target_w = 480

    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
    nodes = torchpipe.pipe({
        "backend": "S[DecodeTensor,PPLWarpAffineTensor ,SyncTensor]",
        "target_h": target_h,
        "target_w": target_w,
        "data_format": "hwc"})

    import numpy as np
    src_points = np.float32([[0, 0],
                            [224 - 1, 0],
                            [442 - 1, 442 - 1]])
    dst_points = np.float32([[0, 0],
                            [333 - 1, 0],
                            [333 - 1, 211 - 1]])
    MM = cv2.getAffineTransform(src_points, dst_points)
    zz_inv = cv2.getAffineTransform(dst_points, src_points)

    cv2_result = cv2.warpAffine(img, MM, (target_w, target_h))

    input = {TASK_DATA_KEY: img_raw,
             "affine_matrix": zz_inv.reshape(6).tolist()}
    nodes(input)

    # img = input["result"].cpu().numpy()
    img = input["result"].cpu().numpy()
    print(img.shape)

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

    test_PPLWarpAffineTensor()
