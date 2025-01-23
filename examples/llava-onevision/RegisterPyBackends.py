
import torchpipe as tp
import torch
storage = tp.ThreadSafeKVStorage.getInstance()

class CSamplingParams:
    def init(self, config):
        print(config)
        return True
    
    def forward(self, input: tp._C.Dict) -> None:
        
        req_id = input['request_id'].decode('utf-8')
        
        token_counter = storage[(req_id, "token_counter")]
        new_tokens = token_counter['new_tokens']
        input_tokens = token_counter['input_tokens']
        sampling_params = input['sampling_params']
        stop_token_ids = sampling_params['stop_token_ids']
        max_tokens = sampling_params['max_tokens']
        max_seq_len = sampling_params['max_seq_len']
        
        generated_token = input['data'].item()
        # import pdb; pdb.set_trace()
        
        print(input.keys())
        result = (input['data'])
        print("generated_token = ", generated_token)
        
        if generated_token in stop_token_ids:
            # eos
            finish_reason = "stop"
        elif max_tokens <= new_tokens:
            input['result'] = generated_token
            finish_reason = "length"
        elif max_seq_len <= input_tokens + new_tokens:
            input['result'] = generated_token
            finish_reason = "length"
        else:
            input['restart'] = 's'
        print("sampling_params = ", sampling_params)
        
from transformers.models.qwen2 import Qwen2Model
prepare_4d_causal_attention_mask_with_cache_position = Qwen2Model._prepare_4d_causal_attention_mask_with_cache_position        
class PrefillCosSinMask:
    def init(self, config):
        print(config)
        self.mask = None
        return True
    
    def forward(self, input: tp._C.Dict) -> None:
        
        req_id = input['request_id'].decode('utf-8')
        
        
        q, k, v  = input['data']
        seq_len = q.shape[-1]
        kv_seq_len = k.shape[-1]
        
        
        attention_mask = torch.ones((1, 1, seq_len, kv_seq_len), dtype=q.dtype, device=q.device)
        cache_position = torch.zeros((1, seq_len), dtype=torch.long)
        batch_size = q.shape[0]


        re = prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
            sequence_length=seq_len,
            target_length=seq_len,
            dtype=q.dtype,
            device=q.device,
            cache_position=cache_position,
            batch_size=batch_size,
            config=None,
            past_key_values=None
        )

        
        print(re)
        
        import pdb; pdb.set_trace()
        
        # import pdb; pdb.set_trace()

# class Pdb:
#     def init(self, config):
#         print(config)
#     def forward(self, input: tp._C.Dict) -> None:
#         import pdb; pdb.set_trace()
# # tp.register_backend(PrefillCosSinMask, "PrefillCosSinMask")
# tp.register_backend(Pdb, "Pdb")

import importlib
module_name='transformers.models.qwen2.modeling_qwen2'
module = importlib.import_module(module_name)
apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb")
repeat_kv = getattr(module, "repeat_kv")
import math
NUM_HEADS=14
HEAD_DIM=64
NUM_KEY_VALUE_HEADS=2
NUM_KEY_VALUE_GROUPS=7
HIDDEN_SIZE=896
def prefill_batchless_forward(query_states, key_states, value_states, cos, sin, attention_mask, past_key = None, past_value = None):
        bsz = 1
        assert query_states.shape[0] == bsz
        q_len = query_states.shape[-2]
        query_states = query_states.view(bsz, q_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2) 
            value_states = torch.cat([past_value, key_states], dim=2) 
            
        past_key, past_value = key_states, value_states

        key_states = repeat_kv(key_states, NUM_KEY_VALUE_GROUPS)
        value_states = repeat_kv(value_states, NUM_KEY_VALUE_GROUPS)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(HEAD_DIM)

        assert attention_mask is not None  # no matter the length, we just slice it
        # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        causal_mask = attention_mask
        # print(causal_mask, attention_mask.shape)
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, NUM_HEADS, q_len, HEAD_DIM):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, NUM_HEADS, q_len, HEAD_DIM)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, HIDDEN_SIZE)
        return attn_output, past_key, past_value
    
def decode_batchless_forward(query_states, key_states, value_states, cos, sin, attention_mask, past_key , past_value):
    bsz = 1
    assert query_states.shape[0] == bsz
    q_len = int(query_states.shape[-2])
    assert q_len == 1
    query_states = query_states.view(bsz, q_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, NUM_KEY_VALUE_HEADS, HEAD_DIM).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    assert past_key is not None
    key_states = torch.cat([past_key, key_states], dim=2) 
    value_states = torch.cat([past_value, value_states], dim=2) 
        
    past_key, past_value = key_states, value_states

    key_states = repeat_kv(key_states, NUM_KEY_VALUE_GROUPS)
    value_states = repeat_kv(value_states, NUM_KEY_VALUE_GROUPS)
    
    

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(HEAD_DIM)

    # print(attn_weights.shape, value_states.shape)
    assert attention_mask is not None  # no matter the length, we just slice it
    # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    causal_mask = attention_mask
    # print(causal_mask, attention_mask.shape)
    attn_weights = attn_weights + causal_mask
        
        

    # upcast attention to fp32
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, NUM_HEADS, q_len, HEAD_DIM):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, NUM_HEADS, q_len, HEAD_DIM)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)
    return attn_output, past_key, past_value

import traceback
class PrefillAttention:
    def init(self, config):
        pass
    def forward(self, input: tp._C.Dict) -> None:
        try:
            query_states, key_states, value_states, cos, sin, attention_mask = input['data']
            re = prefill_batchless_forward(query_states, key_states, value_states, cos, sin, attention_mask)
            # import pdb; pdb.set_trace()
            input['result'] = re
        except Exception as e:
            traceback.print_exc()
            import pdb; pdb.set_trace()

tp.register_backend(PrefillAttention, "PrefillAttention")
class DecodeAttention:
    def init(self, config):
        pass
    def forward(self, input: tp._C.Dict) -> None:
        try:
            query_states, key_states, value_states, cos, sin, attention_mask, past_key, past_value = input['data']
            re = decode_batchless_forward(query_states, key_states, value_states, cos, sin, attention_mask, past_key, past_value)
            # import pdb; pdb.set_trace()
            input['result'] = re
        except Exception as e:
            traceback.print_exc()
            import pdb; pdb.set_trace()

tp.register_backend(DecodeAttention, "DecodeAttention")