# from transformers.models.qwen2.modeling_qwen2 import QWen2ForCausalLM

import os
import types
import torch
from typing import Optional, Tuple, Union
import subprocess
import math
import importlib
from packaging import version
import transformers

try:
    from tqdm import tqdm
except :
    tqdm = lambda x: x
    

def check_if_transformers_greater(target_version: Union[str, version.Version]) -> bool:
    if isinstance(target_version, str):
        target_version = version.parse(target_version)

    return version.parse(transformers.__version__) >= target_version
assert check_if_transformers_greater("4.47.0")

def _modify_attention(attention):
    
    class DummyTorchPluginOp(torch.autograd.Function):
        @staticmethod
        def symbolic(g, q, k, v):
            args = [q, k, v]
            # These become the operator attributes.
            kwargs = {}
            from torch.onnx.symbolic_helper import _get_tensor_sizes
            output_type = q.type().with_sizes(_get_tensor_sizes(q))
            return g.op("CustomTorchOps::TorchPlugin", *args,
                        **kwargs).setType(output_type)

        @staticmethod
        def forward(ctx, q,k,v):
            return q
            
    class DummyTorchPlugin(torch.nn.Module):
        def forward(self, q,k,v):
            x = DummyTorchPluginOp.apply(q,k,v)
            return x


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
  
        # attn_output = query_states # self.batchless_attn(query_states, key_states, value_states)
        attn_output = self.batchless_attn(query_states, key_states, value_states)
        attn_output = self.o_proj(attn_output)
        
 
        return attn_output, None, None
    attention.forward = types.MethodType(forward, attention)
    attention.batchless_attn = DummyTorchPlugin()




 

def _modify_attentions(layers):
    for layer in layers:
        _modify_attention(layer.self_attn)

 



def _modify_model(model):
    def languane_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        
        assert (inputs_embeds is not None)
        
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        for index, decoder_layer in tqdm(enumerate(self.layers)):
            # print(f'decoder_layer {index}')

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=None,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        return (hidden_states, )

    model.forward = types.MethodType(languane_forward, model)


def _modify_model_only_norm(model):
    def languane_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        
        assert (inputs_embeds is not None)
        
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        for index, decoder_layer in tqdm(enumerate(self.layers)):
            # print(f'decoder_layer {index}')
            # import pdb; pdb.set_trace()
            # layer_outputs = decoder_layer(
            #     hidden_states,
            #     attention_mask=None,
            #     position_ids=position_ids,
            #     past_key_value=past_key_values,
            #     output_attentions=output_attentions,
            #     use_cache=use_cache,
            #     cache_position=cache_position,
            #     position_embeddings=None,
            # )

            # hidden_states = layer_outputs[0]
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            q = decoder_layer.self_attn.q_proj(hidden_states)
            k = decoder_layer.self_attn.k_proj(hidden_states)
            v = decoder_layer.self_attn.v_proj(hidden_states)
            return ((q, k, v),)

        # hidden_states = self.norm(hidden_states)
        # return (hidden_states, )

    model.forward = types.MethodType(languane_forward, model)
class EmbedsAsInputWrapper(torch.nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    def forward(self, inputs_embeds, index_select = None):
        with torch.inference_mode():
            out = self.llm.forward(input_ids = None,
                            attention_mask = None,
                            position_ids = None,
                            past_key_values = None,
                            inputs_embeds = inputs_embeds,
                            index_select = index_select)
        
        return out[0]
    
def _modify_causal_llm(llm):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            index_select: torch.LongTensor = None,
        ) -> Tuple:
        
          
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]

        # logits = self.lm_head(hidden_states[-num_logits_to_keep:, :])
        if index_select is not None:
            assert index_select.shape[0] <= hidden_states.shape[0]

            # if len(logits_to_keep_mask.shape) == len(hidden_states.shape) - 1:
            #     logits_to_keep_mask = logits_to_keep_mask.unsqueeze(-1)
            # assert len(logits_to_keep_mask.shape) == len(hidden_states.shape)
            
            # last_dim = int(hidden_states.shape[-1])
            hidden_states = hidden_states.index_select(0, index_select)
            # return tmp
        logits = self.lm_head(hidden_states)
        
        return (logits, )
    llm.forward = types.MethodType(forward, llm)



def _modify_causal_llm_empty(llm):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            index_select: torch.LongTensor = None,
        ) -> Tuple:
        
          
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        return outputs

    llm.forward = types.MethodType(forward, llm)


def export_input_lm_part(model, output_dir, num_layers = -1, use_index_select = True):
    
    # _modify_attentions_empty(model.model.layers) 
    _modify_model_only_norm(model.model)

    _modify_causal_llm_empty(model)
    
    if num_layers is not None and num_layers > 0:
        model.model.layers = model.model.layers[:num_layers]
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "input_lm.onnx")
    
    def export_input_lm(llm):

        # assert past_key_values is not None
        # if return_last_seq_len:
        # use logits_to_keep_mask to reduce the number of logits to keep
        if use_index_select:
            dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'index_select': {0: 'request_size'}}
            index_select = torch.zeros((2,)).to(torch.long).to(llm.model.device)
            index_select[-2] = 2939
            index_select[-1] = 2940
            input_names = ['inputs_embeds', 'index_select']
        else:
            dynamic_axes = {'inputs_embeds': {0: 'seq_len'}}
            input_names = ['inputs_embeds']
        # dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'logits': {0: 'seq_len'}
        
        inputs_embeds = torch.zeros((2941, llm.model.embed_tokens.weight.shape[-1])).to(llm.model.device, torch.float16)
        print(f'start exporting {out_path}')
        torch.onnx.export(EmbedsAsInputWrapper(model),
                        args=(inputs_embeds, index_select),
                        f=out_path,
                        opset_version=17,
                        input_names=input_names,
                        output_names=['q','k','v'],
                        dynamic_axes=dynamic_axes)
        assert 0 == subprocess.call(["onnxsim", out_path, out_path])
        
        # import onnx_graphsurgeon as gs
        # import onnx

        # graph = gs.import_onnx(onnx.load(out_path))

        # for node in graph.nodes:
        #     if node.op == "Reshape":
        #         node.attrs["allowzero"] = 1

        # onnx.save(gs.export_onnx(graph), out_path)

        print(f'input_lm: {out_path} saved.')
    
    export_input_lm(model)
        
def export_batchable_part(model, output_dir, num_layers = -1, use_index_select = True):
    
    _modify_attentions(model.model.layers) 
    _modify_model(model.model)

    _modify_causal_llm(model)
    
    if num_layers is not None and num_layers > 0:
        model.model.layers = model.model.layers[:num_layers]
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "batchable.onnx")
    
    def export_batchable(llm):

        # assert past_key_values is not None
        # if return_last_seq_len:
        # use logits_to_keep_mask to reduce the number of logits to keep
        if use_index_select:
            dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'index_select': {0: 'request_size'}}
            index_select = torch.zeros((2,)).to(torch.long).to(llm.model.device)
            index_select[-2] = 2939
            index_select[-1] = 2940
            input_names = ['inputs_embeds', 'index_select']
        else:
            dynamic_axes = {'inputs_embeds': {0: 'seq_len'}, 'logits': {0: 'seq_len'}}
            input_names = ['inputs_embeds']
        # dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'logits': {0: 'seq_len'}
        
        inputs_embeds = torch.zeros((2941, llm.model.embed_tokens.weight.shape[-1])).to(llm.model.device, torch.float16)
        print(f'start exporting {out_path}')
        torch.onnx.export(EmbedsAsInputWrapper(model),
                        args=(inputs_embeds, index_select),
                        f=out_path,
                        opset_version=17,
                        input_names=input_names,
                        output_names=['logits'],
                        dynamic_axes=dynamic_axes)
        assert 0 == subprocess.call(["onnxsim", out_path, out_path])
        
        # import onnx_graphsurgeon as gs
        # import onnx

        # graph = gs.import_onnx(onnx.load(out_path))

        # for node in graph.nodes:
        #     if node.op == "Reshape":
        #         node.attrs["allowzero"] = 1

        # onnx.save(gs.export_onnx(graph), out_path)

        print(f'Batchable: {out_path} saved.')
    
    export_batchable(model)

def export_batchless_prefill_part(llm, out_dir):

    model = llm.model
    # 获取模块名
    module_name = llm.__module__
    print(f"module_name={module_name}")
    
    # 导入模块
    module = importlib.import_module(module_name)
    apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb")
    repeat_kv = getattr(module, "repeat_kv")

    def batchless_forward(self, query_states, key_states, value_states, cos, sin, attention_mask, past_key = None, past_value = None):
        bsz = 1
        assert query_states.shape[0] == bsz
        q_len = query_states.shape[-2]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2) 
            value_states = torch.cat([past_value, key_states], dim=2) 
            
        past_key, past_value = key_states, value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None  # no matter the length, we just slice it
        # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        causal_mask = attention_mask
        # print(causal_mask, attention_mask.shape)
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return attn_output, past_key, past_value

    model.layers[0].self_attn.forward = types.MethodType(batchless_forward, model.layers[0].self_attn)
    
    bsz = 1
    q_len = 2941
    past_key_values_length = 0
    hidden_size = model.embed_tokens.weight.shape[-1]
    num_heads = model.layers[0].self_attn.num_heads
    num_key_value_heads = model.layers[0].self_attn.num_key_value_heads
    num_key_value_groups = model.layers[0].self_attn.num_key_value_groups
 
    print("hidden_size ", hidden_size, model.layers[0].self_attn.num_heads)
    device = model.device
    query_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    key_states = torch.zeros((bsz, q_len, hidden_size//num_key_value_groups), dtype=torch.float16, device=device)
    value_states = torch.zeros((bsz, q_len, hidden_size//num_key_value_groups), dtype=torch.float16, device=device)
    past_key, past_value = None, None
    cos = torch.zeros((bsz, q_len, hidden_size//num_heads), dtype=torch.float16, device=device)
    sin = torch.zeros((bsz, q_len, hidden_size//num_heads), dtype=torch.float16, device=device)
    attention_mask = torch.zeros((1,1, q_len, q_len), dtype=torch.float16, device=device)

    dynamic_axes={'query_states': {1: 'seq_len'}, 'key_states': {1: 'seq_len'},
                    'value_states': {1: 'seq_len'}, 
                    'attn_output': {1: 'seq_len'}, 'cos': {1: 'seq_len'},
                    'sin': {1: 'seq_len'},'attention_mask': {2: 'seq_len',3: 'seq_len'}}
    dynamic_axes.update({'past_key_output':{2: 'seq_len'}, 'past_value_output':{2: 'seq_len'}})
    out_path = os.path.join(out_dir,"batchless_prefill.onnx")

    torch.onnx.export(model.layers[0].self_attn,
                    args=(query_states,key_states, value_states, cos, sin, attention_mask),
                    f=out_path,
                    opset_version=17,
                    input_names=['query_states','key_states', 'value_states','cos','sin', 'attention_mask'],
                    output_names=['attn_output', 'past_key_output', 'past_value_output'] ,
                    dynamic_axes=dynamic_axes)
    assert 0 == subprocess.call(["onnxsim", out_path, out_path])
    print(f'Prefill Batchless CausalLM: {out_path} saved.')
    


def export_batchless_decode_part(llm, out_dir):

    model = llm.model
    # 获取模块名
    module_name = llm.__module__
    print(f"module_name={module_name}")

    # 导入模块
    module = importlib.import_module(module_name)
    apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb")
    repeat_kv = getattr(module, "repeat_kv")

    def batchless_forward(self, query_states, key_states, value_states, cos, sin, attention_mask, past_key , past_value):
        bsz = 1
        assert query_states.shape[0] == bsz
        q_len = int(query_states.shape[-2])
        assert q_len == 1
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        assert past_key is not None
        key_states = torch.cat([past_key, key_states], dim=2) 
        value_states = torch.cat([past_value, value_states], dim=2) 
            
        past_key, past_value = key_states, value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # print(attn_weights.shape, value_states.shape)
        assert attention_mask is not None  # no matter the length, we just slice it
        # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        causal_mask = attention_mask
        # print(causal_mask, attention_mask.shape)
        attn_weights = attn_weights + causal_mask
            
            

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)
        return attn_output, past_key, past_value

    
    model.layers[0].self_attn.forward = types.MethodType(batchless_forward, model.layers[0].self_attn)
    
    bsz = 1
    q_len = 1
    past = 2941
    q_len_with_past = q_len + past
    
    hidden_size = model.embed_tokens.weight.shape[-1]
    num_heads = model.layers[0].self_attn.num_heads
    num_key_value_heads = model.layers[0].self_attn.num_key_value_heads
    num_key_value_groups = model.layers[0].self_attn.num_key_value_groups
 
    print("hidden_size/num_heads ", hidden_size, num_heads)
    print(f"num_key_value_groups={num_key_value_groups}")
    device = model.device
    query_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    key_states = torch.zeros((bsz, q_len, hidden_size//num_key_value_groups), dtype=torch.float16, device=device)
    value_states = torch.zeros((bsz, q_len, hidden_size//num_key_value_groups), dtype=torch.float16, device=device)
    
    past_key = torch.zeros((1,num_heads//num_key_value_groups, past, hidden_size//num_heads), dtype=torch.float16, device=device)
    past_value = torch.zeros((1,num_heads//num_key_value_groups, past, hidden_size//num_heads), dtype=torch.float16, device=device)
    
    cos = torch.zeros((bsz, q_len, hidden_size//num_heads), dtype=torch.float16, device=device)
    sin = torch.zeros((bsz, q_len, hidden_size//num_heads), dtype=torch.float16, device=device)
    attention_mask = torch.zeros((1,1, q_len, q_len_with_past), dtype=torch.float16, device=device)

    out_path = os.path.join(out_dir,"batchless_decode.onnx")
     
    dynamic_axes={'past_key': {2: 'past'}, 'past_value': {2: 'past'},
                   'attention_mask': {3: 'q_len_with_past'}}
    dynamic_axes.update({'past_key_output':{2: 'q_len_with_past'}, 'past_value_output':{2: 'q_len_with_past'}})

    torch.onnx.export(model.layers[0].self_attn,
                    args=(query_states,key_states, value_states, cos, sin, attention_mask, past_key, past_value),
                    f=out_path,
                    opset_version=17,
                    input_names=['query_states','key_states', 'value_states','cos','sin', 'attention_mask',"past_key","past_value"],
                    output_names=['attn_output', 'past_key_output', 'past_value_output'] ,
                    dynamic_axes=dynamic_axes)
    assert 0 == subprocess.call(["onnxsim", out_path, out_path])
    print(f'Decoding Batchless CausalLM: {out_path} saved.')
    
    
    
