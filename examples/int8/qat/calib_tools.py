# Copyright 2021-2023 NetEase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import calib


except ImportError:
    raise ImportError(
        "pytorch-quantization is not installed. Install from "
        "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
    )

import torch
import os


def _compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()


def _collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in enumerate(data_loader):
        print(f"batch {i}")
        model(image.cuda())
        if i >= num_batches - 1:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


class Calibrator:
    def __init__(self, calib_method="mse", percentile: float = 99.99):
        quant_modules.initialize()

        self.calib_method = calib_method
        self.percentile = percentile

        assert (calib_method in ["max", "mse", "entropy", "percentile"])

        if calib_method != "max":
            calib_method = "histogram"

        quant_desc_input = QuantDescriptor(
            num_bits=8, calib_method=calib_method)

        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)

        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(
            quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

        calibration_per_channel = True
        axis = (0,) if calibration_per_channel else None
        quant_desc_weight = QuantDescriptor(num_bits=8, axis=axis)

        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(
            quant_desc_weight)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

    def calibrate(self, q_model, train_dataloader, num_batches):
        assert (num_batches >= 1)
        q_model.eval()
        with torch.no_grad():
            print(
                f"start collect stats. calib_method = {self.calib_method}, num_batches={num_batches}")
            _collect_stats(q_model, train_dataloader, num_batches=num_batches)
            print("start compute amax")
            _compute_amax(q_model, method=self.calib_method,
                          percentile=self.percentile)

    def __del__(self):
        quant_modules.deactivate()

    def get_torch_nn(self):
        if hasattr(self, "torch_nn"):
            return self.torch_nn

        assert (torch.nn.Conv2d is quant_nn.Conv2d)

        class NN:
            def __init__(self):
                pass

        self.torch_nn = NN()

        for item in quant_modules._DEFAULT_QUANT_MAP:
            setattr(self.torch_nn, item.mod_name,
                    getattr(item.orig_mod, item.mod_name))
        return self.torch_nn


def save_onnx(torch_model, onnx_path, input=None):
    if torch.nn.Conv2d is quant_nn.Conv2d:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
    try:
        if not input is None:
            dummy_input = input.cpu()
        else:
            dummy_input = torch.randn(*(1, 3, 224, 224), device='cpu')

        input_names = ["input"]
        output_names = ["output"]

        torch_model.eval()
        torch_model = torch_model.cpu()

        out_size = len(torch_model(dummy_input))
        out = {"input": {0: "batch_size"}}
        for i in range(out_size):
            out[f"output_{i}"] = {0: "batch_size"}

        torch.onnx.export(torch_model,
                          dummy_input,
                          onnx_path,
                          verbose=False,
                          opset_version=17,
                          do_constant_folding=True,
                          keep_initializers_as_inputs=True,
                          input_names=["input"],      # 输入名
                          output_names=[
                              f"output_{i}" for i in range(out_size)],  # 输出名
                          dynamic_axes=out
                          )
        SIM = True
        if SIM:
            import onnx
            from onnxsim import onnx_simplifier

            onnx_model = onnx.load(onnx_path)
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
            model_simp, check = onnx_simplifier.simplify(onnx_model, check_n=0)
            onnx.save(model_simp, onnx_path)
        print(onnx_path, " saved")
    except Exception as e:
        if torch.nn.Conv2d is quant_nn.Conv2d:
            quant_nn.TensorQuantizer.use_fb_fake_quant = False
        raise e
    if torch.nn.Conv2d is quant_nn.Conv2d:
        quant_nn.TensorQuantizer.use_fb_fake_quant = False


def residual_quantizer():
    return quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)


# todo :https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/qat/lean/quantize.py
