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


from types import MethodType
import os
import struct
import torch
from argparse import ArgumentParser

# fixed batchsize <->dynamic batchsize


def rebatch(infile, outfile, batch_size):
    import onnx

    model = onnx.load(infile)
    graph = model.graph

    # Change batch size in input, output and value_info
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        # tensor.type.tensor_type.shape.dim[0].dim_param = batch_size
        if isinstance(batch_size, int):
            tensor.type.tensor_type.shape.dim[0].dim_value = batch_size
        else:
            tensor.type.tensor_type.shape.dim[0].dim_param = batch_size

    # Set dynamic batch size in reshapes (-1)
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        for init in graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into("q", shape, 0, -1)
                init.raw_data = bytes(shape)

    onnx.save(model, outfile)


def merge_mean_std(raw_onnx, mean, std):
    import onnx

    if isinstance(raw_onnx, bytes):
        onnx_model = onnx.load_model_from_string(raw_onnx)
    else:
        assert os.path.exists(raw_onnx)
        onnx_model = onnx.load(raw_onnx)
    onnx.checker.check_model(onnx_model)
    onnx_model = pre_simp(onnx_model)
    graph = onnx_model.graph

    # print(graph.node[0].op_type)
    # assert(graph.node[0].op_type == "Conv")
    # if graph.node[0].op_type != "Conv":
    #     return onnx._serialize(onnx_model), False
    input_name = "input"
    const_tensors = [x.name for x in graph.initializer]
    for input_node in onnx_model.graph.input:
        if input_node.name in const_tensors:
            continue
        # print(input_node.name, "input_node.name")
        # input_node.name = name
        input_name = input_node.name
        break

    input_shape = input_node.type.tensor_type.shape.dim
    # (x/255 -m )/s == (x -255*m) /  (s*255) == x/ (255s) - m/s
    # if isinstance(mean, list):
    #     mean = [float(x) * -255. for x in mean]
    #     std = [1/(float(x) * 255.) for x in std]
    assert False
    mean = [float(x) * -255.0 for x in mean]
    std = [1 / (float(x) * 255.0) for x in std]
    # print(mean, std)

    # for input_node in onnx_model.graph.input:
    #     input_node.name = name

    index_seconds = []
    for i, input_node in enumerate(graph.node):
        for na_ in input_node.input:
            if na_ == input_name:
                index_seconds.append(i)

    sub_const_node = onnx.helper.make_tensor(
        name="const_sub",
        data_type=onnx.TensorProto.FLOAT,
        dims=[1, len(mean), 1, 1],
        vals=mean,
    )
    # sub_const_node_input = onnx.helper.make_tensor_value_info(name='const_sub',
    #                                          elem_type=onnx.TensorProto.FLOAT,
    #                                          shape=[1, len(mean),1,1])
    graph.initializer.append(sub_const_node)
    # onnx_model.graph.input.append(sub_const_node_input)

    sub_node = onnx.helper.make_node(
        "Add", name="pre_sub", inputs=[input_name, "const_sub"], outputs=["pre_sub"]
    )
    graph.node.insert(0, sub_node)

    # 插入mul
    mul_const_node = onnx.helper.make_tensor(
        name="const_mul",
        data_type=onnx.TensorProto.FLOAT,
        dims=[1, len(std), 1, 1],
        vals=std,
    )
    # mul_const_node_input = onnx.helper.make_tensor_value_info(name='const_mul',
    #                                          elem_type=onnx.TensorProto.FLOAT,
    #                                          shape=[1, len(std),  1,1])
    graph.initializer.append(mul_const_node)
    # onnx_model.graph.input.append(mul_const_node_input)

    # for i, tensor in enumerate(tensors):
    #     value_info = helper.make_tensor_value_info(tensor.name, ONNX_DTYPE[tensor.data_type], tensor.dims)
    #     weight_infos.append(value_info)
    #     graph.input.insert(i+1, value_info) # because 0 is for placeholder, so start index is 1

    mul_node = onnx.helper.make_node(
        "Mul", name="pre_mul", inputs=["pre_sub", "const_mul"], outputs=["pre_mul"]
    )
    graph.node.insert(1, mul_node)

    for index_second in index_seconds:
        # 原本第一层的输入修改 原始版本有bug
        for i, input_node in enumerate(graph.node[2 + index_second].input):
            print(input_node, i)
            if input_name == input_node:
                graph.node[2 + index_second].input[i] = "pre_mul"

    graph = onnx.helper.make_graph(
        graph.node, graph.name, graph.input, graph.output, graph.initializer
    )
    info_model = onnx.helper.make_model(graph)

    add_value_info_for_constants(info_model)
    onnx_model = onnx.shape_inference.infer_shapes(info_model)

    onnx.checker.check_model(onnx_model)
    # print(onnx_model.graph.node[0].op_type) # Add Sub
    # print(onnx_model.graph.node[1].op_type) # Mul

    in_shape = [3]

    for ipt in onnx_model.graph.input:
        if ipt.name == input_name:
            for i, dim in enumerate(ipt.type.tensor_type.shape.dim):
                if i != 0:
                    in_shape.append(dim.dim_value)

    import onnxsim
    from onnxsim import simplify

    # convert model

    model_simp, check = simplify(
        onnx_model, 0, input_shapes={input_name: in_shape}, dynamic_input_shape=True
    )

    if not check:
        print(
            "warning: onnx simplify check failed. Note that the checking is not always correct"
        )
        # onnx.save( model_simp, "a.onnx")
    #
    # print(model_simp.graph.node[0].op_type)
    print("successfully merge_mean_std into onnx")
    return onnx._serialize(model_simp), True


def pre_simp(onnx_model):
    import onnxsim
    from onnxsim import simplify
    import onnx

    onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph

    const_tensors = [x.name for x in graph.initializer]
    print(const_tensors, "const_tensors")

    # for i in graph.node:
    #     print(f"{i.name}, {i.op_type}")

    in_shape = [1]

    name = "input"
    # print(len(onnx_model.graph.input), "len")
    # print((onnx_model.graph.input))

    for input_node in onnx_model.graph.input:
        if input_node.name in const_tensors:
            continue
        # print(input_node.name, "input_node.name")
        # input_node.name = name
        name = input_node.name
    # print(name, const_tensors)
    for ipt in onnx_model.graph.input:
        if ipt.name in const_tensors:
            continue
        name = ipt.name
        for i, dim in enumerate(ipt.type.tensor_type.shape.dim):
            if i != 0:
                in_shape.append(dim.dim_value)
    print(name, in_shape)
    for i in range(len(in_shape)):
        if in_shape[i] == -1:
            in_shape[i] = 480

    cm_vs = compare_version(onnxsim.__version__, "0.4.0")  # 大于0.4.0版本以上，接口改了，需要做选择
    if cm_vs == "lt" or cm_vs == "eq":
        model_simp, check = simplify(
            onnx_model,
            1,
            input_shapes={name: in_shape},
            skip_shape_inference=False,
            dynamic_input_shape=True,
        )
    else:
        model_simp, check = simplify(onnx_model)

    if not check:
        print(
            "warning: onnx simplify check failed. Note that the checking is not always correct"
        )
        # onnx.save( model_simp, "a.onnx")
    onnx.checker.check_model(model_simp)
    return onnx_model


def simp(raw_onnx):
    import onnxsim
    from onnxsim import simplify

    # convert model
    onnx_model = onnx.load(raw_onnx)
    # onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    model_simp = pre_simp(onnx_model)
    # model_simp = onnx.shape_inference.infer_shapes(model_simp)
    return onnx._serialize(model_simp)


def torch2onnx_v0(
    torch_model,
    onnx_save_path,
    input_shape=(3, 224, 224),
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.120, 57.375],
):
    import torch

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

    print("start convert torch`s model to onnx, target is ", onnx_save_path)
    # torch_model = torch_model.eval().to("cpu")
    x = torch.randn(1, *input_shape).to("cuda")
    torch_model.eval()
    out_size = 1
    re = torch_model(x)
    if isinstance(re, (list, tuple)):
        out_size = len(re)
        out = {"input": {0: "batch_size"}}
        for i in range(out_size):
            out[f"output_{i}"] = {0: "batch_size"}
    else:
        out = {"input": {0: "batch_size"}, "output_0": {0: "batch_size"}}  # 批处理变量

    torch.onnx.export(
        torch_model,
        x,
        onnx_save_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],  # 输入名, 只支持单输入
        output_names=[f"output_{i}" for i in range(out_size)],  # 输出名
        dynamic_axes=out,
    )
    print("saved before merge mean and std: ", onnx_save_path)
    onnx_model = onnx.load(onnx_save_path)
    onnx.checker.check_model(onnx_model)
    if not mean:
        assert not std
    else:
        merge(onnx_save_path, onnx_save_path, mean, std)


def forward_function(mean, std):
    mean = torch.as_tensor(mean, dtype=torch.float32, device="cuda")
    std = torch.as_tensor(std, dtype=torch.float32, device="cuda")
    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                dtype
            )
        )

    if max(mean) < 1.00001:
        expand_factor = 255.0
    else:
        expand_factor = 1

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1) * expand_factor
    if std.ndim == 1:
        std = std.view(-1, 1, 1) * expand_factor
        # (x/255 - mean)/std = (x-255*mean)/(255*std)

    def with_mean_std_forward(self, x):
        # print("mean is ", mean, " std is ", std)
        # x = x / 255.
        x = x.sub(mean).div(std)
        y = self._forward(x)
        return y

    return with_mean_std_forward


def torch2onnx(
    torch_model, onnx_save_path, input_shape=(3, 224, 224), mean=None, std=None
):
    import torch

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

    print("start convert torch`s model to onnx, target is ", onnx_save_path)
    # torch_model = torch_model.eval().to("cpu")
    x = torch.randn(1, *input_shape).to("cuda")

    if not mean:
        assert not std
    else:
        fun = forward_function(mean, std)
        torch_model._forward = torch_model.forward
        torch_model.forward = MethodType(fun, torch_model)

    torch_model.eval()
    out_size = 1
    re = torch_model(x)
    if isinstance(re, (list, tuple)):
        out_size = len(re)
        out = {"input": {0: "batch_size"}}
        for i in range(out_size):
            out[f"output_{i}"] = {0: "batch_size"}
    else:
        out = {"input": {0: "batch_size"}, "output_0": {0: "batch_size"}}  # 批处理变量

    torch.onnx.export(
        torch_model,
        x,
        onnx_save_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],  # 输入名, 只支持单输入
        output_names=[f"output_{i}" for i in range(out_size)],  # 输出名
        dynamic_axes=out,
    )
    print("saved  merge mean and std: ", onnx_save_path)
    onnx_model = onnx.load(onnx_save_path)
    onnx.checker.check_model(onnx_model)

    onnx_model = pre_simp(onnx_model)
    onnx.save(onnx_model, onnx_save_path)
    # merge(onnx_save_path, onnx_save_path, mean, std)


def merge(in_path, out_path, mean, std):
    return merge_bn(in_path, out_path, mean, std)


def merge_bn(onnx_input, onnx_output, mean, std):
    onnx_raw = simp(onnx_input)
    onnx_raw, _ = merge_mean_std(onnx_raw, mean, std)

    with open(onnx_output, "wb") as f:
        f.write(onnx_raw)
        print(f"finish {onnx_output}")


def merge_json_mean_and_std(json_path):
    import json5

    with open(json_path, "rb") as f:
        config = json5.load(f)
    for key, value in config.items():
        for k, v in value.items():
            if k == "model_info" and "mean" in value.keys() and "std" in value.keys():
                merge_bn(
                    v,
                    v.replace(".onnx", "_merge_ms.onnx"),
                    value["mean"].split(","),
                    value["std"].split(","),
                )


if __name__ == "__main__":
    merge_json_mean_and_std("/workspace/models/pp_ocr_server.json")

    exit(0)
    input = "../../models/model_ocr_rec_ms.onnx"
    onnx_raw = simp(input)

    out = "../../models/model_ocr_rec_ms2.onnx"
    with open(out, "wb") as f:
        f.write(onnx_raw)
    exit(0)
    with open(input, "rb") as f:
        raw = merge_mean_std(f.read(), [0, 10, 2], [1, 2, 3])
    out = "../../models/model_ocr_det_ms.onnx"
    print(out)
    with open(out, "wb") as f:
        f.write(raw)

    # parser = ArgumentParser('Replace batch size with \'N\'')
    # parser.add_argument('infile')
    # parser.add_argument('outfile')
    # args = parser.parse_args()

    # #rebatch(args.infile, args.outfile, 'N')
    # rebatch(args.infile, args.outfile, 8)


# https://github.com/rohankumardubey/deeplearning4j/blob/810881baaee23ba1aa56fff781e3b315365f2c36/contrib/codegen-tools/onnx-def-gen/convert_model_upgrade.py
# Referenced from: https://github.com/onnx/onnx/issues/2660#issuecomment-605874784
def add_value_info_for_constants(model):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    import onnx
    from onnx import version_converter, helper, ModelProto

    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph: onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)


def summarize_model(input):
    return f"Inputs {len(input.graph.input)} Nodes {len(input.graph.node)} Initializer {len(input.graph.initializer)} Value info {len(input.graph.value_info)}"


def compare_version(v1, v2):
    v1_list = v1.split(".")
    v2_list = v2.split(".")
    v1_len = len(v1_list)
    v2_len = len(v2_list)
    if v1_len > v2_len:
        for i in range(v1_len - v2_len):
            v2_list.append("0")
    elif v2_len > v1_len:
        for i in range(v2_len - v1_len):
            v1_list.append("0")

    for i in range(len(v1_list)):
        if int(v1_list[i]) > int(v2_list[i]):
            return "gt"
        if int(v1_list[i]) < int(v2_list[i]):
            return "lt"
    return "eq"
