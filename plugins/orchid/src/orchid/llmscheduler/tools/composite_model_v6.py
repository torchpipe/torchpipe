from __future__ import annotations

import argparse
import json
import os

import onnx
import onnx.helper as helper


class CompositeModelConverter:
    def __init__(self, model_path: str, output_path: str, config_path: str | None = None, model_id: str | None = None):
        self.model_path = model_path
        self.output_path = output_path

        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        data_path = output_path + "_data"
        if os.path.exists(data_path):
            try:
                os.remove(data_path)
            except OSError:
                pass

        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self._remove_initializers_from_inputs()

        self.input_to_nodes: dict[str, list] = {}
        for n in self.graph.node:
            for inp in n.input:
                self.input_to_nodes.setdefault(inp, []).append(n)

        self.nodes_to_remove: set[str] = set()
        self.new_nodes: list = []
        self.config = self._load_config(config_path, model_id)

    def _remove_initializers_from_inputs(self) -> int:
        init_names = {i.name for i in self.graph.initializer}
        if not init_names:
            return 0
        new_inputs = []
        removed = 0
        for inp in self.graph.input:
            if inp.name in init_names:
                removed += 1
                continue
            new_inputs.append(inp)
        if removed:
            del self.graph.input[:]
            self.graph.input.extend(new_inputs)
        return removed

    def _prune_unused_initializers(self) -> int:
        used = set()
        for n in self.model.graph.node:
            for i in n.input:
                used.add(i)
        keep = []
        removed = 0
        for init in self.model.graph.initializer:
            if init.name in used:
                keep.append(init)
            else:
                removed += 1
        if removed:
            self.model.graph.ClearField("initializer")
            self.model.graph.initializer.extend(keep)
        return removed

    def _load_config(self, config_path: str | None, model_id_arg: str | None) -> dict:
        if not config_path:
            config_path = os.path.join(os.path.dirname(self.model_path), "config.json")

        hf_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                hf_config = json.load(f)

        model_id = model_id_arg
        if not model_id and "_name_or_path" in hf_config:
            model_id = hf_config["_name_or_path"]
        if not model_id:
            model_id = "unknown"

        config: dict[str, object] = {}
        config["model_id"] = model_id
        config["hidden_size"] = hf_config.get("hidden_size")
        config["num_heads"] = hf_config.get("num_attention_heads", 16)
        config["num_kv_heads"] = hf_config.get("num_key_value_heads", config["num_heads"])
        hidden = config.get("hidden_size", 2048) or 2048
        config["head_dim"] = hf_config.get("head_dim", int(hidden) // int(config["num_heads"]))
        config["rope_theta"] = hf_config.get("rope_theta", 1000000.0)
        config["model_type"] = hf_config.get("model_type", "qwen3")
        return config

    def flatten_io(self) -> None:
        input_ids_info = next((inp for inp in self.graph.input if "input_ids" in inp.name), None)
        if input_ids_info:
            input_ids_info.type.tensor_type.shape.dim[0].dim_param = "num_tokens"
            while len(input_ids_info.type.tensor_type.shape.dim) > 1:
                input_ids_info.type.tensor_type.shape.dim.pop()

        for out in self.graph.output:
            if "logits" in out.name or "output" in out.name:
                if len(out.type.tensor_type.shape.dim) >= 2:
                    out.type.tensor_type.shape.dim[0].dim_param = "num_tokens"
                    if len(out.type.tensor_type.shape.dim) == 3:
                        out.type.tensor_type.shape.dim.pop(1)

    def get_consumers(self, tensor_name: str):
        return self.input_to_nodes.get(tensor_name, [])

    def mark_downstream_removal(self, start_tensor: str, stop_node_names: set[str], visited: set[str] | None = None) -> None:
        if visited is None:
            visited = set()
        queue = [start_tensor]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            consumers = self.get_consumers(curr)
            for c in consumers:
                if c.name in stop_node_names:
                    continue
                if c.name not in self.nodes_to_remove:
                    self.nodes_to_remove.add(c.name)
                    for out in c.output:
                        queue.append(out)

    def find_node(self, nodes, patterns):
        for n in nodes:
            for p in patterns:
                if p in n.name:
                    return n
        return None

    def bypass_transpose(self, start_node) -> bool:
        consumers = self.get_consumers(start_node.output[0])
        transpose_node = next((n for n in consumers if n.op_type == "Transpose"), None)
        if not transpose_node:
            return False
        t_consumers = self.get_consumers(transpose_node.output[0])
        for tc in t_consumers:
            for i, inp in enumerate(tc.input):
                if inp == transpose_node.output[0]:
                    tc.input[i] = start_node.output[0]
        self.nodes_to_remove.add(transpose_node.name)
        return True

    def find_norm_output_via_transpose(self, start_node) -> str | None:
        queue = [(start_node, 0)]
        visited = set()
        while queue:
            curr, depth = queue.pop(0)
            if depth > 10:
                continue
            if curr.name in visited:
                continue
            visited.add(curr.name)
            consumers = self.get_consumers(curr.output[0])
            for c in consumers:
                if c.op_type == "Transpose":
                    return c.input[0]
                if c.op_type in ["Cast", "Pow", "Mul", "Add", "Div", "Sqrt", "ReduceMean"]:
                    queue.append((c, depth + 1))
        return None

    def refactor_attention(self) -> None:
        q_patterns = ["q_proj", "query_key_value", "W_pack"]
        k_patterns = ["k_proj"]
        v_patterns = ["v_proj"]
        o_patterns = ["o_proj", "dense"]

        for layer_idx in range(100):
            layer_nodes = [n for n in self.graph.node if f"/layers.{layer_idx}/" in n.name or f"layer.{layer_idx}." in n.name]
            matmuls = [n for n in layer_nodes if n.op_type == "MatMul"]

            q_proj = self.find_node(matmuls, q_patterns)
            k_proj = self.find_node(matmuls, k_patterns)
            v_proj = self.find_node(matmuls, v_patterns)
            o_proj = self.find_node(matmuls, o_patterns)

            if not (q_proj and k_proj and v_proj and o_proj):
                q_proj = self.find_node(layer_nodes, q_patterns)
                k_proj = self.find_node(layer_nodes, k_patterns)
                v_proj = self.find_node(layer_nodes, v_patterns)
                o_proj = self.find_node(layer_nodes, o_patterns)
                if not (q_proj and k_proj and v_proj and o_proj):
                    if layer_idx < 28:
                        continue
                    break

            def fix_reshape_local(node, heads, dim, suffix):
                consumers = self.get_consumers(node.output[0])
                reshape = next((n for n in consumers if n.op_type == "Reshape"), None)
                if not reshape:
                    return None
                shape_name = f"reshape_const_flat_{layer_idx}_{suffix}"
                new_shape = onnx.helper.make_tensor(shape_name, onnx.TensorProto.INT64, [3], [-1, int(heads), int(dim)])
                self.model.graph.initializer.append(new_shape)
                reshape.input[1] = shape_name
                return reshape

            q_reshape = fix_reshape_local(q_proj, self.config["num_heads"], self.config["head_dim"], "q")
            k_reshape = fix_reshape_local(k_proj, self.config["num_kv_heads"], self.config["head_dim"], "k")
            v_reshape = fix_reshape_local(v_proj, self.config["num_kv_heads"], self.config["head_dim"], "v")

            if q_reshape:
                self.bypass_transpose(q_reshape)
            if k_reshape:
                self.bypass_transpose(k_reshape)
            if v_reshape:
                self.bypass_transpose(v_reshape)

            q_norm_out = self.find_norm_output_via_transpose(q_reshape if q_reshape else q_proj)
            k_norm_out = self.find_norm_output_via_transpose(k_reshape if k_reshape else k_proj)

            q_in = q_norm_out if q_norm_out else (q_reshape.output[0] if q_reshape else q_proj.output[0])
            k_in = k_norm_out if k_norm_out else (k_reshape.output[0] if k_reshape else k_proj.output[0])
            v_in = v_reshape.output[0] if v_reshape else v_proj.output[0]

            attn_out_target = o_proj.input[0]
            call_node = helper.make_node(
                "CompositeAttention",
                inputs=[q_in, k_in, v_in],
                outputs=[attn_out_target],
                name=f"/model/layers.{layer_idx}/self_attn/composite_attention",
                domain="com.custom.llm",
                layer_idx=int(layer_idx),
                q_num_heads=int(self.config["num_heads"]),
                kv_num_heads=int(self.config["num_kv_heads"]),
                head_dim=int(self.config["head_dim"]),
                scale=float(1.0 / (float(self.config["head_dim"]) ** 0.5)),
                causal=True,
                softcap=0.0,
                rope_theta=float(self.config["rope_theta"]),
                pos_encoding_mode="ROPE_LLAMA",
                model_id=str(self.config["model_id"]),
                op_schema_version=1,
            )
            self.new_nodes.append(call_node)

            stop_nodes = {o_proj.name}
            self.mark_downstream_removal(q_in, stop_nodes)
            self.mark_downstream_removal(k_in, stop_nodes)
            self.mark_downstream_removal(v_in, stop_nodes)

    def cleanup(self) -> None:
        inputs_to_remove = ["past_key_values", "attention_mask", "position_ids", "cu_seqlens"]
        for inp in self.graph.input:
            for target in inputs_to_remove:
                if target in inp.name:
                    self.mark_downstream_removal(inp.name, set())

        final_nodes = [n for n in self.graph.node if n.name not in self.nodes_to_remove]
        final_nodes.extend(self.new_nodes)

        new_graph_inputs = [inp for inp in self.graph.input if not any(x in inp.name for x in inputs_to_remove)]
        new_graph_outputs = [out for out in self.graph.output if "present" not in out.name]

        used_inputs = set()
        for n in final_nodes:
            for i in n.input:
                used_inputs.add(i)
        new_initializers = [init for init in self.graph.initializer if init.name in used_inputs]

        self.model.graph.ClearField("node")
        self.model.graph.node.extend(final_nodes)
        self.model.graph.ClearField("input")
        self.model.graph.input.extend(new_graph_inputs)
        self.model.graph.ClearField("output")
        self.model.graph.output.extend(new_graph_outputs)
        self.model.graph.ClearField("initializer")
        self.model.graph.initializer.extend(new_initializers)

        has_custom = False
        for o in self.model.opset_import:
            if o.domain == "com.custom.llm":
                has_custom = True
                break
        if not has_custom:
            self.model.opset_import.append(helper.make_opsetid("com.custom.llm", 1))

        while True:
            removed_count = 0
            used_inputs = set()
            for n in self.model.graph.node:
                for i in n.input:
                    used_inputs.add(i)
            new_inits = []
            for init in self.model.graph.initializer:
                if init.name in used_inputs:
                    new_inits.append(init)
                else:
                    removed_count += 1
            if removed_count == 0:
                break
            self.model.graph.ClearField("initializer")
            self.model.graph.initializer.extend(new_inits)

        queue = [out.name for out in self.model.graph.output]
        tensor_to_producer = {}
        for n in self.model.graph.node:
            for out in n.output:
                tensor_to_producer[out] = n
        reachable = set()
        while queue:
            t = queue.pop(0)
            if t in tensor_to_producer:
                prod = tensor_to_producer[t]
                if id(prod) not in reachable:
                    reachable.add(id(prod))
                    for inp in prod.input:
                        queue.append(inp)

        new_nodes = [n for n in self.model.graph.node if id(n) in reachable]
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(new_nodes)
        self.model.graph.ClearField("value_info")
        self._remove_initializers_from_inputs()
        self._prune_unused_initializers()

    def save(self) -> None:
        onnx.save(
            self.model,
            self.output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(self.output_path) + "_data",
            size_threshold=1024,
            convert_attribute=False,
        )

    def run(self) -> None:
        self.flatten_io()
        self.refactor_attention()
        self.cleanup()
        self.save()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX model to use CompositeAttention (v6).")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    args = parser.parse_args(argv)
    CompositeModelConverter(args.model, args.output, args.config, args.model_id).run()


if __name__ == "__main__":
    main()

