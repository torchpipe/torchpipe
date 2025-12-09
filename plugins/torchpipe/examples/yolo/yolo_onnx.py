import os
import subprocess
import requests
import argparse
from typing import Tuple, Optional
import shutil


def download_file(url: str) -> None:
    """
    下载文件到本地路径

    参数:
        url (str): 文件下载链接

    异常:
        RuntimeError: 如果下载失败
    """
    local_path = os.path.basename(url)
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"文件已下载到: {local_path}")
    except Exception as e:
        raise RuntimeError(f"下载文件失败: {str(e)}")


def get_yolo_onnx(
    model_name: Optional[str] = 'yolo11',
    model_size: str = 'n',
    input_shape: Tuple[int, int] = (640, 640),
    batch_size: int = -1,
    opset: int = 17,
) -> str:
    """
    自动安装ultralytics并导出YOLOv8模型到ONNX（支持动态batch和动态输入尺寸）

    参数:
        model_size (str): 模型尺寸，可选['n', 's', 'm', 'l']（对应nano/small/medium/large）
        input_shape (Tuple[int, int]): 输入尺寸（宽, 高），默认(640, 640)，(-1, -1)表示完全动态
        batch_size (int): 批量大小，-1表示动态batch，默认-1
        opset (int): ONNX算子集版本，默认17（推荐≥13）

    返回:
        str: 导出的ONNX模型路径
    """
    # -------------------- 步骤1：自动安装ultralytics --------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        print("检测到未安装ultralytics，正在自动安装...")
        try:
            subprocess.check_call([
                'pip', 'install', '--upgrade', 'ultralytics'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(f"安装ultralytics失败: {str(e)}")

    # -------------------- 步骤2：验证模型尺寸参数 --------------------
    valid_sizes = ['n', 's', 'm', 'l', 'x']
    if model_size not in valid_sizes:
        raise ValueError(f"无效的model_size: {model_size}，可选值: {valid_sizes}")

    # -------------------- 步骤3：下载预训练模型（若本地不存在） --------------------
    new_model_name = f'{model_name}{model_size}'
    if model_name == 'yolo11' or model_name == 'yolo12':
        model_url = f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{new_model_name}.pt'
    else:
        raise ValueError(f"不支持的模型名称: {model_name}，请使用'yolo11'")

    local_model_path = f'{model_name}{model_size}.pt'
    is_downloaded = False

    if not os.path.exists(local_model_path):
        print(f"模型文件不存在，开始下载：{model_url}")
        try:
            response = requests.get(model_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(local_model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("模型下载完成")
            is_downloaded = True
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {str(e)}")

    # -------------------- 步骤4：初始化YOLO模型并导出 --------------------
    try:
        model = YOLO(local_model_path)

        # 配置动态参数
        dynamic_params = {}
        if batch_size == -1:
            dynamic_params['batch'] = True

        # 处理输入形状（支持完全动态）
        if input_shape == (-1, -1):
            dynamic_params['height'] = True
            dynamic_params['width'] = True
            # 使用默认尺寸作为占位符
            imgsz = (640, 640)
        else:
            imgsz = (input_shape[1], input_shape[0])  # 转换为(高, 宽)

        # 执行导出
        onnx_exported = model.export(
            format="onnx",
            imgsz=imgsz,
            batch=batch_size if batch_size != -1 else 1,  # 动态batch时使用1作为占位符
            dynamic=True if dynamic_params else False,
            opset=opset,
            # dynamo=False,
        )

        # 删除下载的.pt文件（如果是本次下载的）
        if True:
            try:
                os.remove(local_model_path)
                print(f"已删除临时模型文件: {local_model_path}")
            except Exception as e:
                print(f"删除模型文件时出错: {str(e)}")

        return onnx_exported

    except Exception as e:
        raise RuntimeError(f"模型导出失败: {str(e)}")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='导出YOLOv8模型到ONNX格式')
    parser.add_argument('--model_name', type=str, default='yolo11',
                        help='模型尺寸: n(nano), s(small), m(medium), l(large)')
    parser.add_argument('--model_size', type=str, default='n',
                        choices=['n', 's', 'm', 'l'],
                        help='模型尺寸: n(nano), s(small), m(medium), l(large)')
    parser.add_argument('--input_width', type=int, default=640,
                        help='输入宽度(-1表示动态，需与高度同时设为-1)')
    parser.add_argument('--input_height', type=int, default=640,
                        help='输入高度(-1表示动态，需与宽度同时设为-1)')
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='批量大小(-1表示动态batch)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX算子集版本')

    args = parser.parse_args()

    # 验证输入形状参数
    if (args.input_width == -1) != (args.input_height == -1):
        raise ValueError("输入宽度和高度必须同时为-1（动态）或同时为正值（固定尺寸）")

    input_shape = (args.input_width, args.input_height)

    # 执行导出
    onnx_path = get_yolo_onnx(
        model_name=args.model_name,
        model_size=args.model_size,
        input_shape=input_shape,
        batch_size=args.batch_size,
        opset=args.opset
    )

    print(f"导出的ONNX模型路径: {onnx_path}")
