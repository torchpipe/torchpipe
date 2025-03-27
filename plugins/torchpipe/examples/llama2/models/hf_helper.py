from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import fire,types
from typing import Optional, Tuple, Callable
def _modify_attention_v0(attention: torch.nn.Module):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    import flashinfer
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.size()
        if q_len > 1:
            pass

        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        
        # The key tensor, shape: [kv_len, num_kv_heads, head_dim_qk] if kv_layout is NHD, 
        # or [num_kv_heads, kv_len, head_dim_qk] if kv_layout is HND.
        if q_len == 1:
            query_states = query_states[0].transpose(0,1)[0]
            attn_output = flashinfer.single_decode_with_kv_cache(
            query_states, key_states[0], value_states[0], 
            kv_layout="HND", 
            use_tensor_cores=False,
            pos_encoding_mode='ROPE_LLAMA') # NONE ROPE_LLAMA
            attn_output = attn_output.view(bsz, q_len, -1)
        else:
            q = query_states.squeeze(0).transpose(0,1)
            attn_output = flashinfer.single_prefill_with_kv_cache(
                q, key_states[0],
                value_states[0], causal=True,
                kv_layout="HND",pos_encoding_mode='ROPE_LLAMA')
            attn_output = attn_output.view(bsz, q_len, -1)
        attn_weights =None
    

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    attention.forward = types.MethodType(forward, attention)


def _modify_attention_v1(attention: torch.nn.Module):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    import flashinfer
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.size()
        if q_len > 1:
            pass

        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)

        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states.transpose(1, 2), value_states.transpose(1, 2), self.layer_idx, cache_kwargs)
            key_states, value_states = key_states.transpose(1, 2).contiguous(), value_states.transpose(1, 2).contiguous()
        
        # The key tensor, shape: [kv_len, num_kv_heads, head_dim_qk] if kv_layout is NHD, 
        # or [num_kv_heads, kv_len, head_dim_qk] if kv_layout is HND.
        if q_len == 1:
            query_states = query_states[0][0]
            attn_output = flashinfer.single_decode_with_kv_cache(
            query_states, key_states[0], value_states[0], 
            kv_layout="NHD", 
            use_tensor_cores=False,
            pos_encoding_mode='ROPE_LLAMA',  # NONE ROPE_LLAMA
            )
            attn_output = attn_output.view(bsz, q_len, -1)
        else:
            q = query_states.squeeze(0)
            if True:
                
                attn_output = flashinfer.single_prefill_with_kv_cache(
                    q, key_states[0],
                    value_states[0], causal=True,
                    kv_layout="NHD",pos_encoding_mode='ROPE_LLAMA',
                    )
            attn_output = attn_output.view(bsz, q_len, -1)
                
        attn_weights = None
    

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    attention.forward = types.MethodType(forward, attention)
    
def _modify_attention_v2(attention: torch.nn.Module):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    import flashinfer
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.size()
        if q_len > 1:
            pass

        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)

        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states.transpose(1, 2), value_states.transpose(1, 2), self.layer_idx, cache_kwargs)
            key_states, value_states = key_states.transpose(1, 2).contiguous(), value_states.transpose(1, 2).contiguous()
        
        # The key tensor, shape: [kv_len, num_kv_heads, head_dim_qk] if kv_layout is NHD, 
        # or [num_kv_heads, kv_len, head_dim_qk] if kv_layout is HND.
        if q_len == 1:
            query_states = query_states[0][0]
            attn_output = flashinfer.single_decode_with_kv_cache(
            query_states, key_states[0], value_states[0], 
            kv_layout="NHD", 
            use_tensor_cores=False,
            pos_encoding_mode='ROPE_LLAMA',  # NONE ROPE_LLAMA
            )
            attn_output = attn_output.view(bsz, q_len, -1)
        else:
            q = query_states.squeeze(0)
            if False:
                
                attn_output = flashinfer.single_prefill_with_kv_cache(
                    q, key_states[0],
                    value_states[0], causal=True,
                    kv_layout="NHD",pos_encoding_mode='ROPE_LLAMA',
                    )
                
            
            else:
                # batch_prefill_with_ragged_kv_cache
                # https://github.com/LinHeLurking/flashinfer/blob/b53a46f8b073e66fbc8fe888e87517b3aea8bd2d/tests/test_batch_prefill_kernels.py#L561
                page_size = 16
                batch_size =1
                qo_len = q_len
                kv_len = key_states.shape[-3]
                num_pages_per_seq = (kv_len + page_size - 1) // page_size
                total_num_pages = num_pages_per_seq * batch_size
                
                num_kv_heads =  key_states.shape[-2]
                kv_shape = [total_num_pages, 2, page_size, num_kv_heads, self.head_dim] # NHD
    
                workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
                wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace_buffer, "NHD"
                )
                
                print(f"qo_len={qo_len}, kv_len={kv_len}, key_states={key_states.shape}")
                q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
                kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
                num_qo_heads = query_states.shape[-2]
                wrapper.plan(
                    q_indptr,
                    kv_indptr,
                    num_qo_heads,
                    num_kv_heads,
                    self.head_dim,
                    causal=True, pos_encoding_mode='ROPE_LLAMA',
                )
                out = torch.empty_like(query_states).squeeze(0)
                attn_output = wrapper.run(q, key_states[0],
                                value_states[0],
                                out=out)
            assert attn_output is out
            assert attn_output.data_ptr() == out.data_ptr() 
            attn_output = out.view(bsz, q_len, -1)
                
        attn_weights = None

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    attention.forward = types.MethodType(forward, attention)

global_kv = None

def set_kv(max_num_pages, num_layers, page_size, num_kv_heads, head_dim):
    global global_kv
    if global_kv is None:
        global_kv = []
        for i in range(num_layers):
            global_kv.append((torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim).half().to(0),
                            torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim).half().to(0)))
def get_kv(layer_idx):
    global global_kv
    return global_kv[layer_idx]
    

    pass
def _modify_attention(attention: torch.nn.Module):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    import flashinfer
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        bsz, q_len, _ = hidden_states.size()
        if q_len > 1:
            pass

        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape)#.transpose(1, 2)

        cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states.transpose(1, 2), value_states.transpose(1, 2), self.layer_idx, cache_kwargs)
            key_states, value_states = key_states.transpose(1, 2).contiguous(), value_states.transpose(1, 2).contiguous()
        
        # The key tensor, shape: [kv_len, num_kv_heads, head_dim_qk] if kv_layout is NHD, 
        # or [num_kv_heads, kv_len, head_dim_qk] if kv_layout is HND.
        if q_len == 1:
            query_states = query_states[0][0]
            attn_output = flashinfer.single_decode_with_kv_cache(
            query_states, key_states[0], value_states[0], 
            kv_layout="NHD", 
            use_tensor_cores=False,
            pos_encoding_mode='ROPE_LLAMA',  # NONE ROPE_LLAMA
            )
            attn_output = attn_output.view(bsz, q_len, -1)
        else:
            q = query_states.squeeze(0)
            if False:
                
                attn_output = flashinfer.single_prefill_with_kv_cache(
                    q, key_states[0],
                    value_states[0], causal=True,
                    kv_layout="NHD",pos_encoding_mode='ROPE_LLAMA',
                    )
                
            
            else:
                # batch_prefill_with_ragged_kv_cache
                # https://github.com/LinHeLurking/flashinfer/blob/b53a46f8b073e66fbc8fe888e87517b3aea8bd2d/tests/test_batch_prefill_kernels.py#L561
                page_size = 16
                batch_size =1
                qo_len = q_len
                kv_len = key_states.shape[-3]
                num_pages_per_seq = (kv_len + page_size - 1) // page_size
                total_num_pages = num_pages_per_seq * batch_size
                
                num_kv_heads =  key_states.shape[-2]
                kv_shape = [total_num_pages, 2, page_size, num_kv_heads, self.head_dim] # NHD
    
                workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
                wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace_buffer, "NHD"
                )
                
                print(f"qo_len={qo_len}, kv_len={kv_len}, key_states={key_states.shape}")
                q_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
                kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
                num_qo_heads = query_states.shape[-2]
                wrapper.plan(
                    q_indptr,
                    kv_indptr,
                    num_qo_heads,
                    num_kv_heads,
                    self.head_dim,
                    causal=True, pos_encoding_mode='ROPE_LLAMA',
                )
                out = torch.empty_like(query_states).squeeze(0)
                attn_output = wrapper.run(q, key_states[0],
                                value_states[0],
                                out=out)
            
            # kvcache
            global global_kv
            page_size = 16
            if global_kv is None:
                max_num_pages = 1000
                set_kv(max_num_pages=max_num_pages, num_layers=32, page_size=16, num_kv_heads=32, head_dim=128)
            k_append = key_states[0]
            v_append = value_states[0]
            paged_kv_cache = global_kv[self.layer_idx]

            nnz_kv = k_append.shape[0]
            # kv_append_indptr, kv_page_indptr, kv_last_page_len
            kv_append_length = torch.tensor([q_len], dtype=torch.int32, device="cuda:0")
            kv_append_indptr = torch.cat([torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]).int()  # [0, 45, 53, 78, 100]

            num_pages_per_req = torch.tensor([1], dtype=torch.int32, device="cuda:0")
            kv_page_indptr = torch.cat([torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]).int()
            kv_last_page_len = torch.tensor([q_len], dtype=torch.int32, device="cuda:0")

            kv_page_indices = torch.arange(1, dtype=torch.int32, device="cuda:0")

            batch_indices, positions = flashinfer.get_batch_indices_positions(
                    kv_append_indptr, flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size), nnz_kv)
            flashinfer.append_paged_kv_cache(
                k_append,
                v_append,
                batch_indices,
                positions,
                paged_kv_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len
            )
            # kvcache

            # https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html
            
            
            assert attn_output is out
            assert attn_output.data_ptr() == out.data_ptr() 
            attn_output = out.view(bsz, q_len, -1)
                
        attn_weights = None

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    attention.forward = types.MethodType(forward, attention)
    
def compute_layer_requirements(config):
    """Calculate memory requirements for base model and per-layer components."""
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    vocab_size = config.vocab_size

    # Base components (embeddings, final norm, output layer)
    base_params = (
        vocab_size * hidden_size +  # token embeddings
        2 * hidden_size +           # final layer norm
        hidden_size * vocab_size    # output layer
    )

    # Per-layer components (attention, MLP, layer norms)
    layer_params = (
        4 * hidden_size**2 +       # attention projections (q,k,v,o)
        3 * hidden_size * intermediate_size +  # MLP layers
        4 * hidden_size             # layer norms (input+post attention)
    )

    return base_params, layer_params


def get_hf_model(model_id='meta-llama/Llama-2-7b-chat-hf', device='cuda', num_layers=None, use_flashinfer=False):
    """Load model with automatic layer adjustment based on available memory."""
    config = AutoConfig.from_pretrained(model_id,
                        trust_remote_code=True )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if num_layers is None and device == "cuda":
        try:
            # Calculate memory requirements
            base_params, layer_params = compute_layer_requirements(config)
            bytes_per_param = 2  # float16 precision
            base_mem = base_params * bytes_per_param
            per_layer_mem = layer_params * bytes_per_param

            # Get available memory
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            free_mem = total_mem - reserved_mem

            # Calculate maximum viable layers
            if free_mem < base_mem:
                raise RuntimeError("Insufficient memory for base components")

            available_for_layers = free_mem - base_mem
            max_layers = min(
                int(available_for_layers // per_layer_mem),
                config.num_hidden_layers
            )
            num_layers = max(max_layers, 1)
            print(f"Automatically selected {num_layers}/{config.num_hidden_layers} layers")
        except Exception as e:
            print(f"Layer adjustment failed: {e}. Using full model.")
            num_layers = config.num_hidden_layers
    else:
        num_layers = num_layers or config.num_hidden_layers

    # Load model with layer truncation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation='eager',
    )
    
    if use_flashinfer:
        for layer in model.model.layers:
            _modify_attention(layer.self_attn)
    
    if num_layers < config.num_hidden_layers:
        model.model.layers = model.model.layers[:num_layers]
        model.config.num_hidden_layers = num_layers

    model.to(device)
    model.eval()
    return model, tokenizer, num_layers


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

def generate_text(model, tokenizer, prompt, max_new_tokens=7):
    """Perform inference with proper resource management."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def inference(model_id='meta-llama/Llama-2-7b-chat-hf',device='cuda' ,num_layers=2, use_flashinfer=True):
    model, tokenizer, num_layers = get_hf_model(model_id, device, num_layers, use_flashinfer=use_flashinfer)
    prompt = "San Francisco is a"
    
    result = generate_text(model, tokenizer, prompt)
    print(f"\nnum_layers = {num_layers}, Generated text: {result}")
    # 2: totalitéaletoreignersbyMSран
    # 32: 

if __name__ == "__main__":
    
    fire.Fire(inference)

    