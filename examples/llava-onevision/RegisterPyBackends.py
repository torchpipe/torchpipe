
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

