# export_and_optimize_onnx.py
import fire
import subprocess
import sys
import importlib
import torch
import timm
import onnx
import os


def install_and_import(package):
    """检查并安装/导入指定的包"""
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing now...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed '{package}'.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install '{package}': {e}")
            sys.exit(1)
        # 重新导入以确保安装成功
        importlib.reload(importlib.import_module('sys'))
        globals()[package] = importlib.import_module(package)


install_and_import('fire')
install_and_import('onnxslim') 
install_and_import('onnxsim')
install_and_import('timm')


try:
    import timm
    import torch
    import onnx
except ImportError as e:
    print(
        f"Critical dependency missing: {e}. Please install PyTorch and timm first.")
    print("Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("         pip install timm")
    sys.exit(1)


def export_and_optimize_model(
    model_name: str,
    output_file: str = None,
    num_classes: int = 3,
    opset: int = 20,
    input_size: tuple = None,
    dynamic_axes: bool = True,
    device: str = 'cpu',
):
    """
    使用 timm 加载模型，导出为 ONNX，并使用 onnxslim 进行优化。

    Args:
        model_name (str): timm 中的模型名称，例如 'resnet18'。
        output_file (str): 输出的原始 ONNX 文件路径。
        num_classes (int, optional): 模型分类的类别数。默认为 1000。
        opset (int, optional): ONNX 的 opset 版本。默认为 11。
        input_size (str, optional): 输入张量的尺寸，格式为 "C,H,W" 或 "N,C,H,W"。
                                    默认为 "3,224,224"。
        dynamic_axes (bool, optional): 是否为批次维度设置动态轴。默认为 True。
        device (str, optional): 运行模型的设备 ('cpu' 或 'cuda')。默认为 'cpu'。

    """
    print(f"Loading model '{model_name}' from timm...")
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )
    model.eval()
    if input_size is None:
        input_size = model.default_cfg.get('input_size', (3, 224, 224))

    print(
        f"Model loaded. Number of classes: {num_classes} input_size={input_size}")

    # 解析输入尺寸
    input_shape_list = [int(x) for x in input_size]
    if len(input_shape_list) == 3:
        c, h, w = input_shape_list
        input_shape = (-1, c, h, w)
        data_shape = (1, c, h, w)
    elif len(input_shape_list) == 4:
        n, c, h, w = input_shape_list
        input_shape = (n, c, h, w)
        data_shape = (abs(n), c, h, w)
    else:
        raise ValueError(
            f"Invalid input_size format: {input_size}. Expected 'C,H,W' or 'N,C,H,W'.")

    print(f"Input shape for export: {input_shape}")

    dummy_input = torch.randn(data_shape).to(device)
    model = model.to(device)
    original_output = model(dummy_input)
    print(f"model output shape: {original_output.shape}")

    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        print("Dynamic axes enabled for 'batch_size' dimension.")

    if output_file is None:
        output_file = f"{model_name}.onnx"
    print(f"Exporting to ONNX with opset {opset}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        dynamo=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_dict,
    )

    print(f"Model '{model_name}' successfully exported to {output_file}")

    # 验证导出的模型
    try:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid.")
    except onnx.checker.ValidationError as e:
        print(f"ONNX model is invalid: {e}")
        return

    # --- ONNXSlim Optimization ---
    print("Starting ONNXSlim optimization...")
    # 如果未指定优化后的输出文件名，则自动生成


    try:
        for _ in range(2):
            subprocess.check_call([
                sys.executable, "-m", "onnxsim", output_file,  output_file
            ])
            subprocess.check_call([
                sys.executable, "-m", "onnxslim", output_file,  output_file
            ])
        
        print(
            f"ONNX model successfully optimized and saved to {output_file}")

        # 验证优化后的模型
        slim_onnx_model = onnx.load(output_file)
        onnx.checker.check_model(slim_onnx_model)
        print("Optimized ONNX model is valid.")

    except Exception as e:
        print(f"Error during ONNXSlim optimization: {e}")
        print(
            f"Original ONNX file {output_file} was created successfully, but optimization failed.")


if __name__ == '__main__':
    fire.Fire(export_and_optimize_model)
