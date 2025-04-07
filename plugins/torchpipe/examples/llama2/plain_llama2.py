import hami
import torchpipe
import torch
import sys, os
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
import flashinfer

from models import hf_helper

max_num_req=10
max_num_page=1000
page_size=16
page_table = hami.default_page_table().init(max_num_req=max_num_req, max_num_page=max_num_page,page_size=page_size)


### -------------  k v cache -------------------------- ##########
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

set_kv(max_num_page, 32, page_size, 32, 128)
### --------------------------------------------------- ########## 

import hami

class Pdb:
    def forward(self, io: List[hami.Dict]):
        data = io[0]
        print(data['request_size'], data['request_id'], data['node_name'])
        print(list(data.keys()))
        data['result'] = data['data']
        # import pdb; pdb.set_trace()
        # print("pdb, d")
hami.register("Pdb", Pdb)



workspace_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda:0") # 1MB
workspace_buffer_decode = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda:0") # 1MB
prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer_decode, "NHD")
     
zero = torch.tensor([0], device='cuda', dtype=torch.int32)

   
class PyPlugin:
    def init(self, params):
        self.params = params
        print(params)
        
        self.layer_idx = int(self.params['layer_idx'])
        
        self.page_size = page_size
        
        PyPlugin.req_ids = None

    def forward(self, io: List[hami.Dict]):
        q, k, v = io[0]['data']
                
        assert q.dtype == torch.float16
        output = io[0]['output'][0]
        
        if self.layer_idx == 0:
            bs, PyPlugin.num_attention_heads, PyPlugin.head_dim = q.shape
            PyPlugin.num_key_value_heads = k.shape[1]
            
            PyPlugin.req_ids, PyPlugin.num_toks = page_table.get_activated()
            print("type(PyPlugin.num_toks)=", type(PyPlugin.num_toks), PyPlugin.num_toks)

            condition = PyPlugin.num_toks > k.shape[0]  # prefill <=, decode >
            PyPlugin.num_decode = int(condition.sum())
            # assume always prefill first
            
            PyPlugin.num_prefill  = len(PyPlugin.req_ids) - PyPlugin.num_decode
            PyPlugin.num_prefill_tok = int(PyPlugin.num_toks[:PyPlugin.num_prefill].sum())

            print(f"activated page_table={page_table.get_activated()}, num_prefill_tok={PyPlugin.num_prefill_tok}, num_decode={PyPlugin.num_decode}", )
        
        # print(f'q.dtype={q.dtype}, layer_idx={self.layer_idx},PyPlugin.num_prefill_tok={PyPlugin.num_prefill_tok}, PyPlugin.num_decode={PyPlugin.num_decode}')
        
        # if PyPlugin.num_prefill_tok > 0 and PyPlugin.num_decode > 0:
        #     assert False
            # todo : side stream
            
        if PyPlugin.num_prefill_tok > 0:
            self.prefill_forward(q[:PyPlugin.num_prefill_tok], k[:PyPlugin.num_prefill_tok], v[:PyPlugin.num_prefill_tok],
                                output[:PyPlugin.num_prefill_tok], PyPlugin.num_prefill, PyPlugin.num_toks[:PyPlugin.num_prefill])
        if PyPlugin.num_decode > 0:
            self.decode_forward(q[PyPlugin.num_prefill_tok:], k[PyPlugin.num_prefill_tok:], v[PyPlugin.num_prefill_tok:],
                                output[PyPlugin.num_prefill_tok:])
    def decode_forward(self, q, k, v, output):
        if self.layer_idx == 0:
            kv_page_indices_np, kv_page_indptr_np, kv_last_page_len_np  = page_table.page_table(PyPlugin.req_ids[PyPlugin.num_prefill_tok:])  
            kv_page_indices = torch.from_numpy(kv_page_indices_np).cuda()
            kv_page_indptr = torch.from_numpy(kv_page_indptr_np).cuda()
            kv_last_page_len = torch.from_numpy(kv_last_page_len_np).cuda()
            print(f"decode: kv_page_indices= {kv_page_indices}, kv_page_indptr {kv_page_indptr}, kv_last_page_len {kv_last_page_len} ")
            
            index_a=kv_page_indices[kv_page_indptr[1:]-1]
            index_b=kv_last_page_len-1
            PyPlugin.decode_page_table = kv_page_indices, kv_page_indptr, kv_last_page_len, kv_page_indices_np, kv_page_indptr_np, kv_last_page_len_np, index_a, index_b
            decode_wrapper.plan(
                    kv_page_indptr,
                    kv_page_indices,
                    kv_last_page_len,
                    PyPlugin.num_attention_heads,
                    PyPlugin.num_key_value_heads,
                    PyPlugin.head_dim,
                    self.page_size,
                    pos_encoding_mode="ROPE_LLAMA",
                    data_type=torch.float16
                )
        else:
            kv_page_indices, kv_page_indptr, kv_last_page_len, kv_page_indices_np, kv_page_indptr_np, kv_last_page_len_np, index_a, index_b = PyPlugin.decode_page_table
        
        paged_kv_cache = global_kv[self.layer_idx]
        # paged_kv_cache[0][999][kv_last_page_len-1] = k
        # paged_kv_cache[1][999][kv_last_page_len-1] = v

        paged_kv_cache[0][index_a, index_b] = k
        paged_kv_cache[1][index_a, index_b] = v
        # todo: https://github.com/NVIDIA/TensorRT-LLM/blob/a2fad51011a48f2cbfee7172047daec74fb0b1b6/tensorrt_llm/_torch/attention_backend/flashinfer.py#L259
        
        decode_wrapper.run(q, paged_kv_cache, out = output)

    def prefill_forward(self, q, k, v, output, num_prefill_tok, num_toks):
        # torch.cuda.synchronize()
        if self.layer_idx == 0:
            PyPlugin.num_toks_gpu = torch.tensor(num_toks, device='cuda')
            cumulative_sum_gpu = torch.cumsum(PyPlugin.num_toks_gpu, dim=0)
            q_indptr = torch.cat((zero, cumulative_sum_gpu), dim=0)

            PyPlugin.kv_indptr = q_indptr
            
            # print(f'before prefill_wrapper plan, q_indptr={q_indptr}, kv_indptr={kv_indptr}, num_attention_heads={PyPlugin.num_attention_heads}, num_key_value_heads={PyPlugin.num_key_value_heads}, head_dim={PyPlugin.head_dim}')
            prefill_wrapper.plan(
                        q_indptr,
                        PyPlugin.kv_indptr,
                        PyPlugin.num_attention_heads,
                        PyPlugin.num_key_value_heads,
                        PyPlugin.head_dim,
                        causal=True, pos_encoding_mode='ROPE_LLAMA',
                    )
            # print(f'prefill output={output.shape}', q.shape, v.shape)
            # torch.cuda.synchronize()

        attn_output = prefill_wrapper.run(q, k,
                                v,
                                out=output)
        assert attn_output is output
        assert attn_output.data_ptr() == output.data_ptr() 
        self.prefill_upadate_kvcache(k, v, PyPlugin.req_ids[:num_prefill_tok], PyPlugin.kv_indptr, PyPlugin.num_toks_gpu)
        
    def prefill_upadate_kvcache(self, k, v, req_ids, kv_indptr, num_toks_gpu):
        paged_kv_cache = global_kv[self.layer_idx]
        
        if self.layer_idx == 0:
            kv_page_indices, kv_page_indptr, kv_last_page_len  = page_table.page_table(req_ids)  
            print(f"kv_page_indices= {kv_page_indices}, kv_page_indptr {kv_page_indptr}, kv_last_page_len {kv_last_page_len} ")
            kv_page_indices = torch.from_numpy(kv_page_indices).cuda()
            kv_page_indptr = torch.from_numpy(kv_page_indptr).cuda()
            kv_last_page_len = torch.from_numpy(kv_last_page_len).cuda()
            
            seq =  num_toks_gpu# flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, 16)
            
            batch_indices, positions = flashinfer.get_batch_indices_positions(kv_indptr, seq, k.shape[0])
            # print(f'seq={seq},positions={positions},batch_indices={batch_indices},k.shape[0]={k.shape[0]}')
            # import pdb; pdb.set_trace()
            
            PyPlugin.prefill_page_table = (batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len)
        else:
            batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len = PyPlugin.prefill_page_table
        flashinfer.append_paged_kv_cache(
            k,
            v,
            batch_indices,
            positions,
            paged_kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len
        )
        # print(kv_page_indptr, kv_last_page_len)

        
if __name__ == '__main__':
    import time
    # time.sleep(10)

    hami.register("TorchPlugin", PyPlugin)
    
    
    model = hami.init_from_file('config/plain_llama2.toml')
    
    exported_params = "./exported_params"
    tokenizer = AutoTokenizer.from_pretrained(exported_params)
    
    # inference
    prompt = "San Francisco is a"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    input_ids = inputs['input_ids']
    # attention_mask = inputs['attention_mask']
    print(inputs, input_ids.shape)
    # print(io)
    io = {'data':input_ids.squeeze(0),"node_name":'embed_token'}

    io[hami.TASK_REQUEST_ID_KEY] = "id"
    io[hami.TASK_MSG_KEY] = hami.TypedDict({"req_tokens": 5,
                                            "max_new_tokens": 7,
                                            "max_tokens":4096})
    model(io)
    
    tokenizer = AutoTokenizer.from_pretrained('exported_params/')
    q = hami.default_queue("net_out")
    result = []
    while not q.empty():
        data = q.get()
        # print(data['data'])
        result+=(data['data'])
    text = tokenizer.decode(result, skip_special_tokens=True)
    print('\n'+prompt+' '+text, '\n')
    # (num_layer = 2) San Francisco is a totalitéaletoreignersbyMSран