import os
import fire


def export_onnx(model_name="resnet101", output_dir="./"):
    """
    导出ONNX模型并转换为TensorRT引擎
    
    Args:
        model_name: 模型名称 (默认: resnet101)
        output_dir: 输出目录 (默认: ./)
    """
    import torch
    import timm

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    if model_name in timm.list_models():
        model = timm.create_model(
            model_name, pretrained=False, exportable=True).eval()
    else:
        raise ValueError(f"model {model_name} not found in timm")

    # 创建示例输入
    x = torch.randn(1, 3, 224, 224)

    # 设置动态batch维度
    dynamic_axes = {
        'input': {0: 'batch_size'},  # 第0维（batch）为动态
        'output': {0: 'batch_size'}
    }

    # 导出ONNX模型
    onnx_path = f"{output_dir}/{model_name}.onnx"
    torch.onnx.export(
        model,
        x,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    print(f"ONNX model saved to {onnx_path}")

    # 使用onnxslim简化模型
    simplified_onnx_path = onnx_path.replace('.onnx', '.onnx')
    os.system(f"onnxslim {onnx_path} {simplified_onnx_path}")
    print(f"Simplified ONNX model saved to {simplified_onnx_path}")
    return
    # 使用trtexec转换模型
    trt_cmd = (
        f"trtexec --onnx={simplified_onnx_path} "
        f"--saveEngine={output_dir}/model_repository/en/resnet_trt/1/model.plan "
        f"--explicitBatch "
        f"--minShapes=input:1x3x224x224 "
        f"--optShapes=input:{batch_size}x3x224x224 "
        f"--maxShapes=input:32x3x224x224 "  # 设置最大batch size为32
        f"--fp16 "
        f"--workspace=2048"
    )

    print(f"Running: {trt_cmd}")
    os.system(trt_cmd)
    print("TensorRT engine conversion completed!")


if __name__ == "__main__":
    fire.Fire(export_onnx)
