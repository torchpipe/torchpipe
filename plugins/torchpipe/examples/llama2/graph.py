import numpy as np
import onnx_graphsurgeon as gs
from typing import Tuple
import torch

# 定义输入形状和填充参数
inp_shape = (1, 3, 22, 22)
pads = [2, 2]

# 创建变量
var_x = gs.Variable(name="x", shape=inp_shape, dtype=np.float32)
var_y = gs.Variable(name="y", dtype=np.float32)

# 创建节点
circ_pad_node = gs.Node(
    name="circ_pad_plugin",
    op="circ_pad_plugin",
    inputs=[var_x],
    outputs=[var_y],
    attrs={"pads": pads, "plugin_namespace": "example"},
)

# 创建图并添加节点
graph = gs.Graph(nodes=[circ_pad_node], inputs=[var_x], outputs=[var_y])

# 清理图
graph.cleanup()

# 保存为 ONNX 文件
onnx_path = "circ_pad_graph.onnx"
import onnx
onnx_model:onnx.ModelProto=gs.export_onnx(graph)
onnx.save(onnx_model, onnx_path)

print(f"Graph saved to {onnx_path}")




import tensorrt.plugin as trtp
import numpy.typing as npt


import tensorrt as trt
import numpy as np
import torch

# 初始化TensorRT组件
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
@trtp.register("my::add_plugin")
def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> Tuple[trtp.TensorDesc]:
    return inp0.like()

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
@trtp.register("example::circ_pad_plugin")
def circ_pad_plugin_desc(
    inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32]
) -> trtp.TensorDesc:
    ndim = inp0.ndim
    out_desc = inp0.like()

    for i in range(np.size(pads) // 2):
        out_desc.shape_expr[ndim - i - 1] += int(
            pads[i * 2] + pads[i * 2 + 1]
        )

    return out_desc


@trtp.impl("example::circ_pad_plugin")
def circ_pad_plugin_impl(
    inp0: trtp.Tensor,
    pads: npt.NDArray[np.int32],
    outputs: Tuple[trtp.Tensor],
    stream: int
) -> None:
    inp_t = torch.as_tensor(inp0, device="cuda")
    out_t = torch.as_tensor(outputs[0], device="cuda")

    out = torch.nn.functional.pad(inp_t, pads.tolist(), mode="circular")
    out_t.copy_(out)
    
    
    

def build_engine(onnx_path):
    # 创建builder和network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 配置builder
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    # 解析ONNX模型
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 构建并返回引擎
    return builder.build_engine_with_config(network, config)

if __name__ == "__main__":
    import tensorrt as trt
    
     

    def list_all_tensorrt_plugins():
        # 获取 TensorRT 插件注册表
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        registry = trt.get_plugin_registry()
        
        # 遍历所有插件创建器
        print("Available TensorRT Plugins:")
        # for i in range(registry.num_creators):
        #     creator = registry.get_creator(i)
        for creator in  registry.all_creators:
            print(f"  - Name: {creator.name}")
            print(f"    Version: {creator.plugin_version}")
            print(f"    Plugin Namespace: {creator.plugin_namespace}")
            print(f"    Field Names: {creator.field_names}\n")

    list_all_tensorrt_plugins()


    # build_engine(onnx_path)
    
   
   
    # cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=11.8 -DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT=/usr/local/cuda-11.8/ -DCMAKE_CUDA_COMPILER_ID=NVIDIA -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain 
    
    
    # cmake .. -DCUDA_VERSION=11.8 -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out ..