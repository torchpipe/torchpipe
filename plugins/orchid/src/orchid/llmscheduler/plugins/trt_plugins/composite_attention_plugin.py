from __future__ import annotations

import os
import numpy as np
import tensorrt as trt
import torch

from orchid.llmscheduler.plugins.composite_attention import composite_attention_impl


class CompositeAttentionPluginCore(trt.IPluginV3OneCore):
    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 0,
        kv_num_heads: int = 0,
        head_dim: int = 0,
        rope_theta: float = 10000.0,
        pos_encoding_mode: int = 0,
    ):
        trt.IPluginV3OneCore.__init__(self)
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.pos_encoding_mode = pos_encoding_mode
        self.plugin_name = "CompositeAttention"
        self.plugin_version = "1"
        self.plugin_namespace = "com.custom.llm"


class CompositeAttentionPluginBuild(trt.IPluginV3OneBuild):
    def __init__(self, core: CompositeAttentionPluginCore):
        trt.IPluginV3OneBuild.__init__(self)
        self.core = core
        self.num_outputs = 1

    def get_nb_outputs(self):
        return 1

    def get_output_data_types(self, input_types):
        return [input_types[0]]

    def get_output_shapes(self, inputs, shape_inputs, expr_builder):
        in_dims = inputs[0]
        nb_dims = len(in_dims)

        if nb_dims == 3:
            dim0 = in_dims[0]
            dim1 = in_dims[1]
            dim2 = in_dims[2]
            new_dim1 = expr_builder.operation(trt.DimensionOperation.PROD, dim1, dim2)
            out_dims = trt.DimsExprs([dim0, new_dim1])
            return [out_dims]

        return [inputs[0]]

    def supports_format_combination(self, pos, in_out, num_inputs):
        dynamic_desc = in_out[pos]
        desc = dynamic_desc.desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos < num_inputs:
            return desc.type == trt.DataType.HALF
        return desc.type == in_out[0].desc.type

    def configure_plugin(self, inp, out):
        pass

    def get_workspace_size(self, inp, out):
        return 0

    def get_valid_tactics(self):
        return []


class CompositeAttentionPluginRuntime(trt.IPluginV3OneRuntime):
    def __init__(self, core: CompositeAttentionPluginCore):
        trt.IPluginV3OneRuntime.__init__(self)
        self.core = core

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        q_ptr = int(inputs[0])
        k_ptr = int(inputs[1])
        v_ptr = int(inputs[2])
        out_ptr = int(outputs[0])

        total_tokens = input_desc[0].dims[0]
        dtype_code_in = int(input_desc[0].type)
        dtype_code_out = int(output_desc[0].type)

        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            composite_attention_impl(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                total_tokens,
                self.core.num_heads,
                self.core.head_dim,
                self.core.kv_num_heads,
                self.core.layer_idx,
                self.core.rope_theta,
                self.core.pos_encoding_mode,
                dtype_code_in,
                dtype_code_out,
            )
        return

    def on_shape_change(self, inp, out):
        pass

    def attach_to_context(self, resource_context):
        return self.core.owner_plugin.clone()

    def get_fields_to_serialize(self):
        return trt.PluginFieldCollection(
            [
                trt.PluginField("layer_idx", np.array([self.core.layer_idx], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("q_num_heads", np.array([self.core.num_heads], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("kv_num_heads", np.array([self.core.kv_num_heads], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("head_dim", np.array([self.core.head_dim], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("rope_theta", np.array([self.core.rope_theta], dtype=np.float32), trt.PluginFieldType.FLOAT32),
                trt.PluginField("pos_encoding_mode", np.array([self.core.pos_encoding_mode], dtype=np.int32), trt.PluginFieldType.INT32),
            ]
        )

    def set_tactic(self, tactic):
        pass


class CompositeAttentionPlugin(trt.IPluginV3):
    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 0,
        kv_num_heads: int = 0,
        head_dim: int = 0,
        rope_theta: float = 10000.0,
        pos_encoding_mode: int = 0,
    ):
        trt.IPluginV3.__init__(self)
        self.core = CompositeAttentionPluginCore(layer_idx, num_heads, kv_num_heads, head_dim, rope_theta, pos_encoding_mode)
        self.core.owner_plugin = self
        self.build = CompositeAttentionPluginBuild(self.core)
        self.runtime = CompositeAttentionPluginRuntime(self.core)

    def get_capability_interface(self, type):
        if type == trt.PluginCapabilityType.CORE:
            return self.core
        if type == trt.PluginCapabilityType.BUILD:
            return self.build
        if type == trt.PluginCapabilityType.RUNTIME:
            return self.runtime
        return None

    def clone(self):
        return CompositeAttentionPlugin(
            self.core.layer_idx,
            self.core.num_heads,
            self.core.kv_num_heads,
            self.core.head_dim,
            self.core.rope_theta,
            self.core.pos_encoding_mode,
        )


class CompositeAttentionPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "CompositeAttention"
        self.plugin_version = "1"
        self.plugin_namespace = "com.custom.llm"
        self.field_names = trt.PluginFieldCollection(
            [
                trt.PluginField("layer_idx", np.array([]), trt.PluginFieldType.INT32),
                trt.PluginField("q_num_heads", np.array([]), trt.PluginFieldType.INT32),
                trt.PluginField("kv_num_heads", np.array([]), trt.PluginFieldType.INT32),
                trt.PluginField("head_dim", np.array([]), trt.PluginFieldType.INT32),
                trt.PluginField("rope_theta", np.array([]), trt.PluginFieldType.FLOAT32),
                trt.PluginField("pos_encoding_mode", np.array([]), trt.PluginFieldType.INT32),
            ]
        )

    def create_plugin(self, name, fc, phase):
        layer_idx = 0
        num_heads = 0
        kv_num_heads = 0
        head_dim = 0
        rope_theta = 10000.0
        pos_encoding_mode = 0

        for field in fc:
            data = field.data

            def get_scalar(v):
                if isinstance(v, np.ndarray):
                    return v.item()
                return v

            if field.name == "layer_idx":
                layer_idx = int(get_scalar(data))
            elif field.name == "q_num_heads":
                num_heads = int(get_scalar(data))
            elif field.name == "kv_num_heads":
                kv_num_heads = int(get_scalar(data))
            elif field.name == "head_dim":
                head_dim = int(get_scalar(data))
            elif field.name == "rope_theta":
                rope_theta = float(get_scalar(data))
            elif field.name == "pos_encoding_mode":
                if field.type == trt.PluginFieldType.INT32:
                    pos_encoding_mode = int(get_scalar(data))
                else:
                    try:
                        s = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                        if "ROPE" in s:
                            pos_encoding_mode = 1
                        else:
                            pos_encoding_mode = int(s)
                    except Exception:
                        pos_encoding_mode = 0

        if rope_theta > 0 and pos_encoding_mode == 0:
            pos_encoding_mode = 1

        return CompositeAttentionPlugin(layer_idx, num_heads, kv_num_heads, head_dim, rope_theta, pos_encoding_mode)


registry = trt.get_plugin_registry()
PLUGIN_CREATOR = CompositeAttentionPluginCreator()
success = registry.register_creator(PLUGIN_CREATOR, "com.custom.llm")
if success:
    print(f"Registered V3 creator 'CompositeAttention': {success}")
