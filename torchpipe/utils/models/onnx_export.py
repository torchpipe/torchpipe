import os
from pprint import pprint
import torch
from typing import Union


def onnx_export(
    model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
    onnx_path=None,
    input=None,
    opset=None,
):
    try:
        import onnxsim
    except ImportError:
        import sys

        command = [sys.executable, "-m", "pip", "install", "onnxsim"]
        print(
            "Installing onnxsim by `",
            " ".join(command),
            "`, please wait for a moment..",
            sep="",
            end="\n",
        )
        import subprocess

        subprocess.check_call(command)
        import onnxsim
    import onnx

    # 如果安装了polygraphy，onnx_graphsurgeon 可以使用更多优化，不然跳过。
    try:
        _is_polygraphy_imported = True
        import onnx_graphsurgeon as gs
        from polygraphy.backend.onnx import fold_constants
    except Exception as e:
        _is_polygraphy_imported = False
        import warnings

        warnings.warn(
            f"polygraphy and onnx_graphsurgeon are not installed, some optimization will not be used. You can use `pip install polygraphy onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com ` to install them."
        )
    if opset is None:
        import torchpipe
        try:
            opset = torchpipe.supported_opset()
            assert opset > 10 and opset < 100
        except Exception as e:
            opset = None

    
    if input is None:
        dummy_input = torch.randn(*(1, 3, 224, 224), device="cpu")

    elif (isinstance(input, tuple) and isinstance(input[0], int)) or isinstance(input, dict):
        dummy_input = torch.randn(*input, device="cpu")
    elif (isinstance(input, tuple) and isinstance(input[0], torch.Tensor)):
        raise ValueError("multiple input not support now")
        dummy_input = input
    elif isinstance(input, torch.Tensor):
        dummy_input = input.cpu()
    else:
        raise ValueError("input type error")

    model = model.eval()

    input_names = []
    output_names = []

    # input
    out = {"input": {0: "batch_size"}}
    input_names.append("input")
    # output
    if isinstance(dummy_input, torch.Tensor):
        out_size = len(model(dummy_input))
    elif isinstance(dummy_input, tuple):
        out_size = len(model(*dummy_input))
    else:
        raise ValueError("input type error")
    for i in range(out_size):
        out[f"output_{i}"] = {0: "batch_size"}
        output_names.append(f"output_{i}")

    if onnx_path is None:
        import tempfile

        onnx_path = os.path.join(
            tempfile.gettempdir(), f"{model.__class__.__name__}.onnx"
        )
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        keep_initializers_as_inputs=True,
        input_names=input_names,  # 输入名
        output_names=output_names,  # 输出名
        dynamic_axes=out,
    )
    print(f"saved onnx : {onnx_path} ")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    if _is_polygraphy_imported:
        graph = gs.import_onnx(onnx_model)
        print(
            f"Original .. {len(graph.nodes)} nodes, {len(graph.tensors().keys())} tensors, {len(graph.inputs)} inputs, {len(graph.outputs)} outputs"
        )
        graph.cleanup().toposort()
        onnx_graph = fold_constants(
            gs.export_onnx(graph), allow_onnxruntime_shape_inference=True
        )

        onnx_graph = onnx.shape_inference.infer_shapes(onnx_graph)
        graph = gs.import_onnx(onnx_graph)

        graph.cleanup().toposort()
        onnx_model = gs.export_onnx(graph)

    from onnxsim import onnx_simplifier

    # onnx_model = onnx.shape_inference.infer_shapes(onnx_model) ## swin会有bug
    model_simp, check = onnx_simplifier.simplify(onnx_model, check_n=0)
    if not check:
        print(
            "warning: onnx simplify check failed. Note that the checking is not always correct"
        )

    onnx.save(model_simp, onnx_path)
    print(f"simplify over : {onnx_path}  ")


def onnx_run(onnx_path, input):
    import onnxruntime as ort
    import numpy as np

    ort_sess = ort.InferenceSession(onnx_path)

    # process input type
    if isinstance(input, torch.Tensor):
        input = input.cpu().numpy()
    elif isinstance(input, np.ndarray):
        input = input
    else:
        raise ValueError("input type error")

    # run
    outputs = ort_sess.run(None, {"input": input.astype("float32")})

    return outputs


if __name__ == "__main__":
    import timm

    tmp_dir = "/tmp"
    model_name = "resnet50"

    m = timm.create_model(model_name, pretrained=True,
                          exportable=True, num_classes=3)
    m.eval()
    onnx_path = os.path.join(tmp_dir, f"{model_name}.onnx")
    export_onnx(m, onnx_path)
