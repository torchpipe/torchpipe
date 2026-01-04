from torchpipe import pipe
import torch

from torchvision.models.resnet import resnet18

# create some regular pytorch model...
model = resnet18(pretrained=True).eval().cuda()

# create example model
model_path = f"./resnet18.onnx"
x = torch.ones((1, 3, 224, 224)).cuda()
torch.onnx.export(model, x, model_path, opset_version=17, dynamo=False,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

thread_safe_pipe = pipe({
    "preprocessor": {
        "backend": "S[DecodeTensor,ResizeTensor,CvtColorTensor,SyncTensor]",
        'instance_num': 2,
        'color': 'rgb',
        'resize_h': '448',
        'resize_w': '448',
        'next': 'model',

    },
    "model": {
        "backend": "SyncTensor[TensorrtTensor]",
        "model": model_path,
        "model::cache": model_path.replace(".onnx", ".trt"),
        "max": '4',
        'batching_timeout': 4,  # ms, timeout for batching
        'instance_num': 2,
        'mean': "123.675, 116.28, 103.53",
        'std': "58.395, 57.120, 57.375",  # merged into trt
    }}
)
