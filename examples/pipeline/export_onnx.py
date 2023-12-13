


from  torchpipe.utils.models import onnx_export

import os, onnx


def export_onnx_resnet18():
    import torchvision
    model = torchvision.models.resnet18(pretrained=True)
    onnx_export(model, "resnet18.onnx", None, 17)
    import onnxsim
    for i in range(3):
        model_simp, check = onnxsim.simplify("resnet18.onnx")
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, "resnet18.onnx")

    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 --shapes=input:16x3x224x224  --saveEngine=resnet18_16.trt
    #  18.8828 0.2
    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:32x3x224x224 --maxShapes=input:32x3x224x224 --shapes=input:32x3x224x224  --saveEngine=resnet18_32.trt
    #  18.8828 0.2
    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:8x3x224x224 --shapes=input:8x3x224x224  --saveEngine=resnet18_8.trt
    #  18.8828 0.2
    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:4x3x224x224 --shapes=input:4x3x224x224  --saveEngine=resnet18_4.trt
    #  18.8828 0.2
    # trtexec --onnx=resnet18.onnx --fp16  --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:



from ultralytics import YOLO
import torch
from copy import deepcopy
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder


def export_yolov8(model_path="yolov8n.pt", input_shape=(1,3,640,640)):
    model = YOLO(model_path).model
    model = deepcopy(model)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()


    for m in model.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):  # Segment and Pose use Detect base class
            m.dynamic = True
            m.export = True
            m.format = 'onnx'
        elif isinstance(m, C2f):
            m.forward = m.forward_split

    input = torch.randn(input_shape)
    for _ in range(2):
        model(input) 

    onnx_path = os.path.basename(model_path).replace(".pt", ".onnx")

    torch.onnx.export(
        model,
        input,
        onnx_path,
        verbose=False,
        opset_version=16,
        input_names=["images"],
        output_names=["pred"],
        dynamic_axes={
            "images": {0: "batch"},
            "pred": {0: "batch", 2: "anchors"},
        },
    )
    import onnxsim
    for i in range(3):
        model_simp, check = onnxsim.simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, onnx_path)

    print(f"{onnx_path} saved")
     

if __name__ == '__main__':

    export_onnx_resnet18()

    import tempfile
    
    model_name = "yolov8n.pt"
    target = os.path.join(tempfile.gettempdir(), model_name)
    
    if not os.path.exists(target):
        os.system(f"wget -O {target} https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}")

    export_yolov8(model_path=target, input_shape=(1,3,640,640))
     