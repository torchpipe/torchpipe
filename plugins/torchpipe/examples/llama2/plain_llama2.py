import hami
import torchpipe
import torch
import sys, os
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
import flashinfer

from models import hf_helper

max_num_req=20
page_size=16
num_layers = 2

page_table = None
def get_page_table(max_num_page=4096//page_size):
    if page_table is None:
        page_table = hami.default_page_table().init(max_num_req=max_num_req, max_num_page=max_num_page,page_size=page_size)
        set_kv(max_num_page, num_layers, page_size, 32, 128)
    return page_table

### -------------  k v cache -------------------------- ##########
global_kv = None
def set_kv(max_num_pages, num_layers, page_size, num_kv_heads, head_dim):
    global global_kv
    if global_kv is None:
        global_kv = []
        for i in range(num_layers):
            global_kv.append((torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim).half().cuda(),
                            torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim).half().cuda()))
def get_kv(layer_idx):
    global global_kv
    return global_kv[layer_idx]

current_memory = torch.cuda.memory_allocated()  
print(f"当前显存占用: {current_memory / 1024**2:.2f} MB")


cached_memory = torch.cuda.memory_reserved()  
print(f"缓存显存: {cached_memory / 1024**2:.2f} MB")

### --------------------------------------------------- ########## 

import hami

class Pdb:
    def forward(self, io: List[hami.Dict]):
        data = io[0]

        print("Pdb: data.keys()", list(data.keys()), data['request_size'], data['request_id'], data['node_name'], data['data'])
        data['result'] = data['data']
        # import pdb; pdb.set_trace()
        # print("pdb, d")
hami.register("Pdb", Pdb)



workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0") # 1MB
workspace_buffer_decode = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0") # 1MB
prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer_decode, "NHD")
     
zero = torch.tensor([0], device='cuda', dtype=torch.int32)

   
class PyPlugin:
    def init(self, params):
        self.params = params
        print('params: ', params)
        
        self.layer_idx = int(self.params['layer_idx'])
        
        self.page_size = page_size
        
        PyPlugin.req_ids = None

    def forward(self, io: List[hami.Dict]):
        try:
            q, k, v = io[0]['data']
                    
            assert q.dtype == torch.float16
            output = io[0]['output'][0]
            
            if self.layer_idx == 0:
                bs, PyPlugin.num_attention_heads, PyPlugin.head_dim = q.shape
                PyPlugin.num_key_value_heads = k.shape[1]
                
                PyPlugin.req_ids, PyPlugin.num_toks = page_table.pop_activated()
                prefill_size = page_table.get_prefill_num_req_toks(PyPlugin.req_ids)
                print(" (PyPlugin.num_toks)=",  (PyPlugin.req_ids), PyPlugin.num_toks,q.shape, k.shape, prefill_size)

                is_prefill = PyPlugin.num_toks > prefill_size  # prefill <=, decode >
                PyPlugin.num_decode = int(is_prefill.sum())
                # assume always prefill first
                
                PyPlugin.num_prefill  = len(PyPlugin.req_ids) - PyPlugin.num_decode
                PyPlugin.num_prefill_tok = int(PyPlugin.num_toks[:PyPlugin.num_prefill].sum())

                print(f"num_prefill={PyPlugin.num_prefill}. num_prefill_tok={PyPlugin.num_prefill_tok}, num_decode={PyPlugin.num_decode}, req_ids {PyPlugin.req_ids}, num_toks = {PyPlugin.num_toks} ", )
            
            # print(f'q.dtype={q.dtype}, layer_idx={self.layer_idx},PyPlugin.num_prefill_tok={PyPlugin.num_prefill_tok}, PyPlugin.num_decode={PyPlugin.num_decode}')
            
            # if PyPlugin.num_prefill_tok > 0 and PyPlugin.num_decode > 0:
            #     assert False
                # todo : side stream
                
            if PyPlugin.num_prefill_tok > 0:
                
                self.prefill_forward(q[:PyPlugin.num_prefill_tok], k[:PyPlugin.num_prefill_tok], v[:PyPlugin.num_prefill_tok],
                                    output[:PyPlugin.num_prefill_tok], PyPlugin.num_toks[:PyPlugin.num_prefill])
            if PyPlugin.num_decode > 0:
                # return # todo
                self.decode_forward(q[PyPlugin.num_prefill_tok:], k[PyPlugin.num_prefill_tok:], v[PyPlugin.num_prefill_tok:],
                                    output[PyPlugin.num_prefill_tok:])
        except Exception as e:
            print(f"error {e}", flush=True)
            raise e
    def decode_forward(self, q, k, v, output):
        if self.layer_idx == 0:
            kv_page_indices_np, kv_page_indptr_np, kv_last_page_len_np  = page_table.page_table(PyPlugin.req_ids[PyPlugin.num_prefill:])  
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
        print(f"index_a, index_b = {index_a}, {index_b} layer_idx = {self.layer_idx}")
        paged_kv_cache[0][index_a, index_b] = k
        paged_kv_cache[1][index_a, index_b] = v
        # todo: https://github.com/NVIDIA/TensorRT-LLM/blob/a2fad51011a48f2cbfee7172047daec74fb0b1b6/tensorrt_llm/_torch/attention_backend/flashinfer.py#L259
        
        decode_wrapper.run(q, paged_kv_cache, out = output)

    def prefill_forward(self, q, k, v, output, num_toks):
        # torch.cuda.synchronize()
        if self.layer_idx == 0:
            PyPlugin.num_toks_gpu = torch.tensor(num_toks, device='cuda')
            cumulative_sum_gpu = torch.cumsum(PyPlugin.num_toks_gpu, dim=0)
            q_indptr = torch.cat((zero, cumulative_sum_gpu), dim=0)#.contiguous()

            PyPlugin.kv_indptr = q_indptr
            
            print(f'before prefill_wrapper plan, q_indptr={q_indptr}, num_attention_heads={PyPlugin.num_attention_heads}, num_key_value_heads={PyPlugin.num_key_value_heads}, head_dim={PyPlugin.head_dim}')
            prefill_wrapper.plan(
                        q_indptr,
                        PyPlugin.kv_indptr,
                        PyPlugin.num_attention_heads,
                        PyPlugin.num_key_value_heads,
                        PyPlugin.head_dim,
                        causal=True, pos_encoding_mode='ROPE_LLAMA',
                    )
            
        print(output.shape, q.shape, k.shape)
        attn_output = prefill_wrapper.run(q, k,
                                v,
                                out=output)
        assert attn_output is output
        assert attn_output.data_ptr() == output.data_ptr() 
        
        # torch.cuda.synchronize()
        
        # if self.layer_idx == 1:
        #     print(f'prefill output={torch.mean(output)}', output.shape, v.shape)
        
        self.prefill_upadate_kvcache(k, v, PyPlugin.req_ids[:PyPlugin.num_prefill], PyPlugin.kv_indptr, PyPlugin.num_toks_gpu)
        
    def prefill_upadate_kvcache(self, k, v, req_ids, kv_indptr, num_toks_gpu):
        paged_kv_cache = global_kv[self.layer_idx]
        
        if self.layer_idx == 0:
            kv_page_indices, kv_page_indptr, kv_last_page_len  = page_table.page_table(req_ids)  
            print(f"req_ids={req_ids} kv_page_indices= {kv_page_indices}, kv_page_indptr {kv_page_indptr}, kv_last_page_len {kv_last_page_len} ")
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
    prompt = "San Francisco is a totalitéaletoreignersbyMSран"
    prompt = "San Francisco is a"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    input_ids = inputs['input_ids']
    # attention_mask = inputs['attention_mask']
    print(f"inputs = {input_ids}")
    # print(io)
    ios = []
    ids = []
    events = []
    for i in range(20):
        ids.append(f"id-{i}")
        max_tokens = 700
        if i == 5: 
            max_tokens = 10
        if i == 2: 
            max_tokens = 4
        if i == 0:
            max_tokens += 12
        max_tokens = 128
             
        io = hami.Dict({'data':input_ids.squeeze(0),"node_name":'embed_token'})
        io[hami.TASK_EVENT_KEY] = hami.Event() 
        # events.append(io.set_event())
        events.append(io[hami.TASK_EVENT_KEY])
        io[hami.TASK_REQUEST_ID_KEY] = f"id-{i}"
        io[hami.TASK_MSG_KEY] = hami.TypedDict({"req_tokens": input_ids.size(-1),
                                                "max_tokens": max_tokens,
                                                "context_length":4096})
        ios.append(io)
    model(ios)
    
    for ev in events:
        ev.wait()
        
    tokenizer = AutoTokenizer.from_pretrained('exported_params/')
    q = hami.default_queue("net_out")
    
    results = {}
    for id in ids:
        results[id] = []
    while not q.empty():
        data = q.get()
        id = data['request_id']
        results[id] += data['data']
    for key, result in results.items():
        finish_reason = results.pop("finish_reason", None) 
        text = tokenizer.decode(result, skip_special_tokens=True)
        print(f'\n {key}: '+prompt+' '+text + f"-{finish_reason}")
    # (num_layer = 2) San Francisco is a totalitéaletoreignersbyMSран
    # 22777|totalité, 9457|alet, 13606|oreign, 414|ers, 1609|by 
    #  id-0: San Francisco is a totalitéketting器 AußerTaggedahnpinningzza tailmente Selonroid Wars Riv Transkoids‏ fingerprintű Kirk Ind fresoca Einzeln AußeroustacheHDovisuality assemb Bedeut array subsidiariesilleurspeciesumm sweiore":{"inceculptronectorypidglassesтетani forthems Commonwealthvie Razmenteairesonicaciesume virtuel Profildorfjes pingazon swe inspirationning wid
    
    # for key, result in results.items():
    #     text = ""
    #     for index, item in enumerate(result):
    #         text += f' {index}| '+tokenizer.decode(item, skip_special_tokens=True)
    #     print(f'\n {key}: '+prompt+' '+text, '\n')
        
    
    