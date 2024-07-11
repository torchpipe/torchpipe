 
from transformers.models.llama import   LlamaConfig, LlamaForCausalLM # LlamaAttention
from transformers.models.llama import  modeling_llama
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os

#Attention layers whose sequence length dimension cannot be batched
class BatchlessAttention(nn.Module):
    # from typing import ClassVar
    # __constants__ = ["kernel_shape", "strides", "pads", "group", 'z']
    __constants__ = []
    # # Attributes to match the plugin requirements.
    # # Must follow the type annotations via PEP 526-style.
    # # https://peps.python.org/pep-0526/#class-and-instance-variable-annotations
    # kernel_shape: ClassVar[List[int]]
    # strides: ClassVar[List[int]]
    # pads: ClassVar[List[int]]
    # group: int
    # z: int
     
    def __init__(self):
        super().__init__()
        
    def forward(self, query_states, key_states, value_states):
        # shape should be keeped
        # see: https://github.com/leimao/TensorRT-Custom-Plugin-Example/issues/8
        return query_states # + key_states + value_states

#Attention layers whose sequence length dimension can be batched
if False:
    # put the following in modeling_llama.py
    class BatchfulAttention(LlamaAttention):
        """
        Llama Batchful attention module. This module inherits from `LlamaAttention` as the weights of the module stays
        untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
        flash attention and deal with padding tokens in case the input contains any of them.
        """
        def __init__(self, config: LlamaConfig):
            super().__init__(config=config)

            self.batchless_attn = BatchlessAttention()
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            seqlens_in_batch: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            # LlamaFlashAttention2 attention does not support output_attentions
            if "padding_mask" in kwargs:
                warnings.warn(
                    "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
                )

                # overwrite attention_mask with padding_mask
                attention_mask = kwargs.pop("padding_mask")

            output_attentions = False
            assert(len(hidden_states.size()) == 2)
            q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # Flash attention requires the input to have the shape
            # batch_size x seq_length x head_dim x hidden_dim
            # therefore we just need to keep the original shape
            # query_states = query_states.view(q_len, self.num_heads, self.head_dim).transpose(0,1)
            # key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim).transpose(0,1)
            # value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim).transpose(0,1)

            # custom plugin
            attn_output = self.batchless_attn(query_states, key_states, value_states)
            
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, None, None
        

def export_decode_batchful(llm: LlamaForCausalLM, out_dir = 'onnx/'):
    class Wrapper(torch.nn.Module):
        def __init__(self, llm):
            super().__init__()
            self.llm = llm
        def forward(self, inputs_embeds):
            out = self.llm.forward(input_ids = None,
                            attention_mask = None,
                            position_ids = None,
                            past_key_values = None,
                            inputs_embeds = inputs_embeds)
           
            print('out', out[1:])
            return out[0]
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # return next_tokens
    model = Wrapper(llm).eval()

    # assert past_key_values is not None
    
    dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'logits': {0: 'seq_len'}}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "./decode_batchful.onnx")

    inputs_embeds = torch.zeros((244, 2560)).to(model.llm.device, torch.float16)
    
    torch.onnx.export(model,
                      args=(inputs_embeds,),
                      f=out_path,
                      opset_version=17,
                      input_names=['inputs_embeds'],
                      output_names=['logits'] ,
                      dynamic_axes=dynamic_axes,
                      export_modules_as_functions={modeling_llama.BatchlessAttention})
    print(f'{out_path} saved with custom BatchlessAttention.')


def export_vila_prefilling(llm: LlamaForCausalLM, out_dir : str  = 'onnx/'):
    class Wrapper(torch.nn.Module):
        def __init__(self, llm):
            super().__init__()
            self.llm = llm
        def forward(self, inputs_embeds, position_ids):
            out = self.llm.forward(input_ids = None,
                            attention_mask = None,
                            position_ids = position_ids,
                            past_key_values = None,
                            inputs_embeds = inputs_embeds,
                            use_cache=True)
            present_key_value = out[1]
           
            print('out', out[2:])
            return out[0], present_key_value
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # return next_tokens
    model = Wrapper(llm).eval()

    # assert past_key_values is not None
    
    dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'logits': {0: 'seq_len'}}
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prefilling.onnx")

    past_key_values_length = 0
    seq_length = 244
    inputs_embeds = torch.zeros((1, seq_length, 2560)).to(model.llm.device, torch.float16)
    position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=model.llm.device
            )
    position_ids = position_ids.unsqueeze(0)
    
    
    
    
    # past =  torch.zeros((1, 20, seq_length, 128)).to(model.llm.device, torch.float16)
    # present_key_value = [(past, past) for _ in range(32)]
    present_key_value_names = [(f"past_key_{i}", f"past_vaule_{i}") for i in range(32)]
    present_key_value_names = [item for sublist in present_key_value_names for item in sublist]
    present_key_value_dynamic = {item:{2:'seq_len'} for item in present_key_value_names }
    dynamic_axes={'inputs_embeds': {1: 'seq_len'}, 'logits': {1: 'seq_len'},
                  'position_ids': {1: 'seq_len'}}
    dynamic_axes.update(present_key_value_dynamic)
    print("dynamic_axes: ", dynamic_axes)
    # exit(0)
    
    torch.onnx.export(model,
                      args=(inputs_embeds, position_ids),
                      f=out_path,
                      opset_version=17,
                      input_names=['inputs_embeds', 'position_ids'],
                      output_names=['logits'] + present_key_value_names ,
                      dynamic_axes=dynamic_axes)
    print(f'{out_path} saved prefilling.')




def self_attn_forward(self, query_states, key_states, value_states, position_ids, past_key, past_value):
        # shape should be keeped
        # temporary impl. would be replaced by custom onnx/tensorrt plugin
        # see: https://github.com/leimao/TensorRT-Custom-Plugin-Example/issues/8

        print(type(self))
        bsz, q_len, _ = query_states.size()
        assert(bsz == 1)
        bsz = 1
        attention_mask = None
        past_key_value = (past_key, past_value) if past_key is not None else None
        
        if past_key is not None: # decoding stage
            q_len = 1

        query_states = query_states.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(1, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(1, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2] # 
        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) 

        key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
        value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        return attn_output, key_states, value_states
    

def export_vila_decode_batchless(self_attn):
    import types
    self_attn.forward = types.MethodType(self_attn_forward, self_attn)
    
    bsz = 1
    q_len = 1
    
    past_q_len = 244
    
    device = torch.device("cuda")
    query_states = torch.zeros((bsz, q_len, 2560), dtype=torch.float16, device=device)
    key_states = torch.zeros((bsz, q_len, 2560), dtype=torch.float16, device=device)
    value_states = torch.zeros((bsz, q_len, 2560), dtype=torch.float16, device=device)
    
    position_ids = torch.arange(
                    past_q_len, q_len + past_q_len, dtype=torch.long, device=device
                ).unsqueeze(0)
    past_key = torch.zeros((bsz, 20, past_q_len, 128), dtype=torch.float16, device=device)
    past_value = torch.zeros((bsz, 20, past_q_len, 128), dtype=torch.float16, device=device)
    
    dynamic_axes={'past_key': {2: 'past_seq_len'}, 'past_value': {2: 'past_seq_len'}}
    
    out_path = "onnx/batchless.onnx"

    torch.onnx.export(self_attn,
                      args=(query_states,key_states, value_states,position_ids,past_key,past_value),
                      f=out_path,
                      opset_version=17,
                      input_names=['query_states','key_states', 'value_states','position_ids','past_key','past_value'],
                      output_names=['attn_output', 'past_key_output', 'past_value_output'] ,
                      dynamic_axes=dynamic_axes)
    print(f'{out_path} saved with custom export_vila_decode_batchless.')


def export_prefill_batchless(self_attn, is_prefill=True):
    import types
    self_attn.forward = types.MethodType(self_attn_forward, self_attn)
    
    device = torch.device("cuda")
    bsz = 1
    if is_prefill:
        q_len = 244
        past_key_values_length = 0
        query_states = torch.zeros((bsz, q_len, 2560), dtype=torch.float16, device=device)
        key_states = torch.zeros((bsz, q_len, 2560), dtype=torch.float16, device=device)
        value_states = torch.zeros((bsz, q_len, 2560), dtype=torch.float16, device=device)
        past_key, past_value = None, None

        position_ids = torch.arange(
                    past_key_values_length, q_len + past_key_values_length, dtype=torch.long, device=device
                ).unsqueeze(0)

        present_key_value_names = [(f"past_key_{i}", f"past_vaule_{i}") for i in range(32)]
        present_key_value_names = [item for sublist in present_key_value_names for item in sublist]
        present_key_value_dynamic = {item: {2:'seq_len'} for item in present_key_value_names}
        dynamic_axes={'query_states': {1: 'seq_len'}, 'key_states': {1: 'seq_len'},
                      'value_states': {1: 'seq_len'}, 
                      'attn_output': {1: 'seq_len'}, 'position_ids': {1: 'seq_len'}}
        dynamic_axes.update(present_key_value_dynamic)
        out_path = "onnx/batchless_prefill.onnx"

        torch.onnx.export(self_attn,
                        args=(query_states,key_states, value_states,position_ids, past_key, past_value,),
                        f=out_path,
                        opset_version=17,
                        input_names=['query_states','key_states', 'value_states','position_ids'],
                        output_names=['attn_output', 'past_key_output', 'past_value_output'] ,
                        dynamic_axes=dynamic_axes)
        print(f'{out_path} saved with custom export_vila_decode_batchless.')

    else:
        assert False, 'Not checked yet, use export_vila_decode_batchless instead'
        q_len = 1
        
        past_q_len = 244
        
        query_states = torch.zeros((q_len, 2560), dtype=torch.float16, device=device)
        key_states = torch.zeros((q_len, 2560), dtype=torch.float16, device=device)
        value_states = torch.zeros((q_len, 2560), dtype=torch.float16, device=device)
        position_ids = torch.ones((1, q_len), dtype=torch.long, device=device)*past_q_len
        past_key = torch.zeros((q_len, 20, past_q_len, 128), dtype=torch.float16, device=device)
        past_value = torch.zeros((q_len, 20, past_q_len, 128), dtype=torch.float16, device=device)
        
        dynamic_axes={'past_key': {2: 'past_seq_len'}, 'past_value': {2: 'past_seq_len'}}
    
        out_path = "onnx/batchless.onnx"

        torch.onnx.export(self_attn,
                        args=(query_states,key_states, value_states,position_ids,past_key,past_value),
                        f=out_path,
                        opset_version=17,
                        input_names=['query_states','key_states', 'value_states','position_ids','past_key','past_value'],
                        output_names=['attn_output', 'past_key_output', 'past_value_output'] ,
                        dynamic_axes=dynamic_axes)
        print(f'{out_path} saved with custom export_vila_decode_batchless.')

 
    
