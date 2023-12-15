import torchpipe

 
import cv2
import torch
import pytest

@pytest.mark.skipif(not torchpipe.libipipe.is_registered('PPLResizeTensor'), reason="PPLResizeTensor backend is not registered")
def test_PPLResizeTensor():
    import numpy as np
    img_path = "assets/image/gray.jpg"
    img_raw=open(img_path,'rb').read()
    img=cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
    print(img.shape)#(274, 442, 3)
    resize_h= 224
    resize_w= 480
    
    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
    nodes =  torchpipe.pipe({
    "backend":"S[DecodeTensor,PPLResizeTensor ,SyncTensor]",
    "resize_h":resize_h,
    "resize_w":resize_w,
    "data_format":"hwc"})

    import numpy as np
 
    cv2_result = cv2.resize(img,(resize_w, resize_h))
    
    input = {TASK_DATA_KEY: img_raw}
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
    assert(mean_diff<1.2)


@pytest.mark.skipif(not torchpipe.libipipe.is_registered('PPLResizeCenterPadTensor'), reason="PPLResizeCenterPadTensor backend is not registered")
def test_PPLResizeCenterPadTensor():
    import numpy as np
    img_path = "assets/image/gray.jpg"
    img_raw=open(img_path,'rb').read()
    img=cv2.imdecode(np.frombuffer(img_raw, np.uint8), cv2.IMREAD_COLOR)
    print(img.shape)#(274, 442, 3)
    ori_img = img
    resize_h= 224
    resize_w= 480
    
    from torchpipe import pipe, TASK_DATA_KEY, TASK_RESULT_KEY
    nodes =  torchpipe.pipe({
    "backend":"S[DecodeTensor,PPLResizeCenterPadTensor ,SyncTensor]",
    "resize_h":resize_h,
    "resize_w":resize_w,
    "data_format":"hwc"})

    import numpy as np
 
    cv2_result = cv2.resize(img,(resize_w, resize_h))
    
    input = {TASK_DATA_KEY: img_raw}
    nodes(input)
 
    # img = input["result"].cpu().numpy()

    img_h = ori_img.shape[0]
    img_w = ori_img.shape[1]
    ratio = max(img_h/resize_h, img_w/resize_w)

    img = input["result"].cpu().numpy()
    print(img.shape, input['top'],input['left'],input['ratio'], ratio)

    
    assert(abs(input['ratio']- ratio)<1e-6 )

    assert(input['top'] == int((resize_h -img_h/ratio)/2))
    assert(input['left'] == int((resize_w - img_w/ratio)/2))    
    

    # cv2.imwrite("debug_cv2.jpg",cv2_result)
    # cv2.imwrite("debug.jpg",img)
    # diff = cv2.absdiff(img, cv2_result)
    # non_zero_diff = diff[np.nonzero(diff)]
    # mean_diff = np.mean(np.abs(non_zero_diff))
    # print(mean_diff)
    # assert(mean_diff<1.2)
 
if __name__ == "__main__":
    import time
    # time.sleep(8)
    
    test_PPLResizeCenterPadTensor()