

import ctypes
import os
import glob
import numpy as np
import torchpipe
from torch.utils import cpp_extension

import torch

import omniback
import cv2


cpp = torch.utils.cpp_extension.load(
    name="yolo_cpp_extension",
    sources=["yolo.cpp"],
    extra_cflags=["-O3", "-Wall", "-std=c++17"],
    extra_include_paths=[]+omniback.libinfo.include_paths(),
    extra_ldflags=[f"-L{omniback.get_library_dir()}", '-lomniback'],
    verbose=True,
    is_python_module=True,
)


def main(
    model: str = "yolo",
    weights: str = "yolo11m.onnx",
    source: str = "https://ultralytics.com/images/bus.jpg",
    imgsz: int = 320,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 1000,
):
    config = {
        'jpg_decode': {
            'backend': 'S[DecodeMat,LetterBoxMat,CvtColorMat,Mat2Tensor,SyncTensor]',
            'resize_h': 320,
            'resize_w': 320,
            'color': 'rgb',
            'instance_num': 6,
            'next': 'model'},
        'model': {'backend': 'S[TensorrtTensor,SyncTensor]',
                  "model": weights,
                  'model::cache': weights.replace('.onnx', '320.trt'),
                  'max': '4x3x320x320',
                  'instance_num': 2,
                  'std': '255,255,255', }
    }

    model = omniback.pipe(config)
    import requests
    from PIL import Image
    from io import BytesIO
    response = requests.get(source)
    # read the bytes
    img_bytes = BytesIO(response.content).read()
    io = {'data': img_bytes}
    model(io)
    print(io.keys())

    # postprocess results in python
    offset_x, offset_y = io['offset']
    scale = io['scale']
    net_predict = io['result']

    boxes = cpp.yolo11_post(
        net_predict, conf_thres=conf_thres, iou_thres=iou_thres)

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    for box in boxes:
        x1, y1, x2, y2, label, prob = box.x1, box.y1, box.x2, box.y2, box.id, box.score
        x1 = (x1 - offset_x)/scale
        y1 = (y1 - offset_y)/scale
        x2 = (x2 - offset_x)/scale
        y2 = (y2 - offset_y)/scale
        print(x1, y1, x2, y2, label, prob)

        label += 1
        color = (int(label * 64 % 256), int(label * 128 %
                 256), int(label * 192 % 256))
        img = cv2.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(
                y2)), color, 2
        )
        cv2.putText(img, str(label-1), (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    out_path = "dog_result.jpg"
    cv2.imwrite(out_path, img[:, :, ::1])
    print(f"result saved in: {out_path} ; There are {len(boxes)} boxes.")

    # print(re)
    # draw boxes on the image


if __name__ == "__main__":
    import fire
    fire.Fire(main)
