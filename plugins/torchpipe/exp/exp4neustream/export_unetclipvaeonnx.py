# from diffusers import StableDiffusionPipeline
import sys
import os
import time
from PIL import Image
from modules import ClipModule, UNetModule, VaeModule, SafetyModule
from torch import nn
import torch
import json
from typing import Dict
device = "cuda"


def export_dynamic_onnx(model, fwd_method,
                        onnx_path: str,
                        net_shapes: tuple | list,
                        dtypes: torch.dtype | list[torch.dtype],
                        input_names: list[str] = None,
                        output_names: list[str] = None,
                        export_params: bool = True,
                        save_as_external_data=True) -> str:
    """
    导出支持动态维度的ONNX模型（支持多输入）
    
    Args:
        model: PyTorch模型
        fwd_method: 前向方法
        onnx_path: 输出路径
        net_shapes: 示例输入形状列表，用-1表示动态维度
                    如 [(-1,77), (-1,3,224,-1)] 或单个形状 (-1,77)
        dtypes: 对应每个输入的数据类型（单个或列表）
        input_names: 输入节点名称列表（默认自动生成）
        output_names: 输出节点名称列表（默认自动生成）
        export_params: 是否导出模型参数
    """
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            return fwd_method(*args)  # 支持多参数输入

    model = WrappedModel(model)
    model.eval()

    def import_or_install(pkg):
        try:
            __import__(pkg)
        except ImportError:
            import subprocess
            subprocess.run(['pip', 'install', pkg], check=True)

    # 安装依赖
    for pkg in ['onnxslim', 'onnx']:
        import_or_install(pkg)

    device = next(model.parameters()).device

    # 统一输入格式处理
    if isinstance(net_shapes[0], int):  # 单输入情况
        net_shapes = [net_shapes]
    if isinstance(dtypes, torch.dtype):  # 单数据类型情况
        dtypes = [dtypes] * len(net_shapes)

    # 创建示例输入张量
    dummy_inputs = [
        torch.ones(*[1 if d == -1 else d for d in net_shape],
                   dtype=dtype
                   ).to(device) for net_shape, dtype in zip(net_shapes, dtypes)]

    # 构建dynamic_axes配置
    dynamic_axes = {}
    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(net_shapes))]

    for name, net_shape in zip(input_names, net_shapes):
        dynamic_axes[name] = {}
        for dim, size in enumerate(net_shape):
            if size == -1:
                dynamic_axes[name][dim] = f"dim{dim}"

    # 设置默认输出名称
    if output_names is None:
        output_names = [f"output_{i}" for i in range(
            len(fwd_method(*dummy_inputs)))]

    # 导出ONNX模型
    torch.onnx.export(
        model,
        tuple(dummy_inputs),  # 转换为元组
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        export_params=export_params,
    )

    # 模型优化
    import onnxslim
    onnxslim.slim(onnx_path, onnx_path,
                  save_as_external_data=save_as_external_data)

    return onnx_path


# exit(0)
sys.path.insert(0, './')


class StableDiffusionPipeline():
    def __init__(self, config_path, **kwargs):
        super().__init__()
        fp = open(config_path, "r")
        config = json.load(fp)
        self.stream_module_list = []
        print(config.keys())

        self.modules = {'ClipModule': ClipModule,
                        'UNetModule': UNetModule,
                        'VaeModule': VaeModule,
                        'SafetyModule': SafetyModule}

        for key, value in config.items():
            value['instance_index'] = 0
            self.modules[key] = self.modules[key](**value)
            self.modules[key].deploy()

    def default_deploy(self, **kwargs):
        for module in self.stream_module_list:
            module.deploy()


torch.set_grad_enabled(False)

# init pipeline from config
sd_config_file = "stable_diffusion_v1_5/config.json"
sd_pipeline = StableDiffusionPipeline(config_path=sd_config_file)

image_size = 256

warm_up_request = {
    "prompt": "a beautiful girl studying in Chinese University",
    "height": image_size,
    "width": image_size,
    "loop_num": {
        "UNetModule": 50
    },
    "request_time": time.time(),
    "guidance_scale": 7.5,
    "seed": 0,
    "SLO": 10000,
    "loop_index": {
        "UNetModule": 0
    },
    "id": -1
}

ONNX_EXPORT = ['clip', 'unet', 'safety', 'vae']
# ONNX_EXPORT = ['safety']
if 'clip' in ONNX_EXPORT:
    clip_model = sd_pipeline.modules['ClipModule'].text_encoder

    def forward(input):
        return clip_model(input, attention_mask=None)[0]
    export_dynamic_onnx(clip_model, forward, 'clip.onnx',
                        (-1, 77), dtypes=torch.long, export_params=True)

    # /root/autodl-tmp/TensorRT/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/bin/trtexec --onnx=clip.onnx --fp16 --shapes=input:20x77
if 'unet' in ONNX_EXPORT:
    unet_model = sd_pipeline.modules['UNetModule'].unet
    # latent_model_input=torch.Size([2, 4, 32, 32]) torch.Size([1]) torch.Size([2, 77, 768])

    def forward(latent_model_input, timestamps, prompt_embeds):
        # [bs, 2, 4, 32, 32] -> [2*bs, 4, 32, 32]
        # latent_model_input = latent_model_input.reshape(-1, 4, 32, 32)
        # latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
        latent_model_input = latent_model_input.repeat(2, 1, 1, 1)
        prompt_embeds = prompt_embeds.reshape(-1, 77, 768)
        result = unet_model(latent_model_input, timestamps,
                            encoder_hidden_states=prompt_embeds).sample
        # print(result)
        return result.reshape(-1, 2, 4, 32, 32)

    export_dynamic_onnx(unet_model, forward, 'unet.onnx', [
                        (-1, 4, 32, 32), (-1,), (-1, 2, 77, 768)], dtypes=torch.float16, export_params=True)
    # (bs1) 5.30081 (bs4) 10.6521-2.51  (bs8)18.0301-2.25375  （bs12）24.8342-2.06 (bs16)34.4774-2.1
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/autodl-tmp/TensorRT/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/lib/
    #  /root/autodl-tmp/TensorRT/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/bin/trtexec --onnx=unet.onnx --fp16 --shapes=input_0:1x2x4x32x32,input_1:1,input_2:1x2x77x768
    #  /root/autodl-tmp/TensorRT/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/bin/trtexec --onnx=unet.onnx --saveEngine=tmp.trt --fp16 --shapes=input_0:4x2x4x32x32,input_1:4,input_2:4x2x77x768 --profilingVerbosity=detailed
    #  /root/autodl-tmp/TensorRT/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/bin/trtexec --onnx=unet.onnx --fp16 --shapes=input_0:8x2x4x32x32,input_1:8,input_2:8x2x77x768
    # /root/autodl-tmp/TensorRT/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/bin/trtexec --onnx=unet.onnx --fp16 --shapes=input_0:12x2x4x32x32,input_1:12,input_2:12x2x77x768
    # /root/autodl-tmp/TensorRT/TensorRT-10.9.0.34/targets/x86_64-linux-gnu/bin/trtexec --onnx=unet.onnx --fp16 --shapes=input_0:16x2x4x32x32,input_1:16,input_2:16x2x77x768
    #
    # trtexec --onnx=unet.onnx \
    #     --saveEngine=tmp0.trt \
    #     --fp16 \
    #     --minShapes=input_0:1x2x4x32x32,input_1:1,input_2:1x2x77x768 \
    #     --optShapes=input_0:8x2x4x32x32,input_1:8,input_2:8x2x77x768 \
    #     --maxShapes=input_0:8x2x4x32x32,input_1:8,input_2:8x2x77x768 --shapes=input_0:1x2x4x32x32,input_1:1,input_2:1x2x77x768

    # trtexec  --onnx=unet.onnx --saveEngine=tmp0.trt --fp16 --shapes=input_0:4x2x4x32x32,input_1:4,input_2:4x2x77x768
    #  --useCudaGraph
    # trtexec --onnx=unet.onnx --fp16 --shapes=input_0:4x2x4x32x32,input_1:4,input_2:4x2x77x768 --saveEngine=tmp.trt --noDataTransfers  --profilingVerbosity=detailed --dumpLayerInfo --dumpProfile --separateProfileRun > trace.json
 #  trtexec --onnx=unet.onnx --fp16 --shapes=input_0:1x2x4x32x32,input_1:1,input_2:1x2x77x768 --loadEngine=tmp.trt  --exportTimes=trace.json
if 'vae' in ONNX_EXPORT:
    vae_model = sd_pipeline.modules['VaeModule'].vae
    scaling_factor = vae_model.config.scaling_factor
    # latent_model_input=torch.Size([2, 4, 32, 32]) torch.Size([1]) torch.Size([2, 77, 768])

    def forward(latents):
        # [bs, 2, 4, 32, 32] -> [2*bs, 4, 32, 32]
        # latent_model_input = latent_model_input.reshape(-1, 4, 32, 32)
        # prompt_embeds = prompt_embeds.reshape(-1, 77, 768)
        images = vae_model.decode(
            latents / scaling_factor, return_dict=False)[0]
        return images

    export_dynamic_onnx(vae_model, forward, 'vae.onnx',
                        (-1, 4, 32, 32), dtypes=torch.float16, export_params=True)


if 'safety' in ONNX_EXPORT:
    safety_model = sd_pipeline.modules['SafetyModule'].safety_checker

    def forward(pixel_values):
        images = torch.zeros(pixel_values.shape[0], 3, 256, 256)
        has_nsfw_concept = safety_model.forward_onnx(
            images=images, clip_input=pixel_values)[1]
        return has_nsfw_concept

    export_dynamic_onnx(safety_model, forward, 'safety.onnx', (-1, 3, 224, 224),
                        dtypes=torch.float16, export_params=True, save_as_external_data=True)

print(warm_up_request.keys())
sd_pipeline.modules['ClipModule'].compute([warm_up_request])
print(warm_up_request.keys())
for i in range(warm_up_request["loop_num"]['UNetModule']):
    sd_pipeline.modules['UNetModule'].compute([warm_up_request])
    warm_up_request["loop_index"]['UNetModule'] += 1
    # print( warm_up_request["loop_index"]['UNetModule'])

print('UNetModule result:', warm_up_request.keys())

sd_pipeline.modules['VaeModule'].compute([warm_up_request])
print('VaeModule result:', warm_up_request.keys())
sd_pipeline.modules['SafetyModule'].compute([warm_up_request])
print(warm_up_request.keys())

img = warm_up_request['pillow_image']
# print(warm_up_request)
img.save("output_image.jpg", quality=95)  # quality控制质量（1-100）
print(f'saved to output_image.jpg')
