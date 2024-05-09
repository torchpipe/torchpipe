# import torchpipe as tp
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    dest="model",
    type=str,
    default="resnet101",
    help="model name",
)
args = parser.parse_args()


def export_onnx(onnx_save_path, model_name):
    import torch
    import torchvision.models as models
    import timm

    # resnet101 = tp.utils.models.create_model(name).eval()
    if model_name in timm.list_models():
        model = timm.create_model(model_name, pretrained=False, exportable=True).eval()
    else:
        if "faster_vit" in model_name:
            assert timm.__version__ == "0.9.6"
            import fastervit

            # timm==0.9.6

            model = fastervit.create_model(model_name, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    if not onnx_save_path:
        onnx_save_path = f"./{model_name}.onnx"
    tp.utils.models.onnx_export(model, onnx_save_path, x)


if __name__ == "__main__":
    model_name = args.model
    op = f"./{model_name}.onnx"
    export_onnx(op, model_name)
    os.system(
        f"trtexec --onnx={op} --saveEngine=./model_repository/en/resnet_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:16x3x224x224 --maxShapes=input:16x3x224x224 --fp16"
    )
