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

    torch.onnx.export(
        model,
        input,
        "yolov8n.onnx",
        verbose=False,
        opset_version=16,
        input_names=["images"],
        output_names=["pred"],
        dynamic_axes={
            "images": {0: "batch"},
            "pred": {0: "batch", 2: "anchors"},
        },
    )
if __name__ == "__main__":
    export_yolov8(model_path="yolov8n.pt", input_shape=(1,3,640,640))