import hami
import torchpipe
import torch
import sys, os
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
import flashinfer

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
# from models.partial_hf import get_hf_model
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

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
class PrefillPlugin:
    def init(self, params):
        self.params = params
        self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
    def forward(self, io: List[hami.Dict]):
        q, k, v = io[0]['data']
        output = io[0]['output'][0]
        
class PyPlugin:
    def init(self, params):
        self.params = params
        print(params)
        
        self.layer_idx = int(self.params['layer_idx'])

        self.tensor_page_table = hami.init("StreamGuard[TensorPage]")#.as_function()
        # self.side_stream =  hami.init("TaskLoop[SideStreamTensor]").as_function()
        # self.side_stream = hami.init("BackendgroundThread[]")
        self.prefill = PrefillPlugin()
        self.prefill.init(params)
        self.num_prefill_tok = 0
        self.num_decode = 0
        self.num_attention_heads = 0
        self.head_dim = 0
        self.num_key_value_heads =0
        
        self.page_size = page_size
        
        self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        workspace_buffer, "NHD")
        self.req_ids = None

    def forward(self, io: List[hami.Dict]):
        q, k, v = io[0]['data']
        print(f'q.dtype={q.dtype}, layer_idx={self.layer_idx}')
        assert q.dtype == torch.float16
        output = io[0]['output'][0]
        
        if self.layer_idx == 0:
            bs, self.num_attention_heads, self.head_dim = q.shape
            self.num_key_value_heads = k.shape[1]
            
            self.req_ids, num_toks = page_table.get_activated()
            print("type(num_toks)=", type(num_toks), num_toks)
            # num_prefill_tok = np.argmax(num_toks > k.shape[0])
            condition = num_toks > k.shape[0]  # prefill <=, decode >
            self.num_decode = int(condition.sum())
            # assume always prefill first
            
            self.num_prefill  = len(self.req_ids) - self.num_decode
            self.num_prefill_tok = int(num_toks[:self.num_prefill].sum())

            print(f"activated page_table={page_table.get_activated()}, num_prefill_tok={self.num_prefill_tok}, num_decode={self.num_decode}", )
        
        if self.num_prefill_tok > 0 and self.num_decode > 0:
            pass
        elif self.num_prefill_tok > 0:
            self.prefill_forward(q[:self.num_prefill_tok], k[:self.num_prefill_tok], v[:self.num_prefill_tok],
                                output[:self.num_prefill_tok], self.num_prefill, num_toks[:self.num_prefill])
        elif self.num_decode > 0:
            self.decode_forward(q[self.num_prefill_tok:], k[self.num_prefill_tok:], v[self.num_prefill_tok:],
                                output[self.num_prefill_tok:], self.num_decode, num_toks[self.num_prefill:])
            
    def decode_forward(self, q, k, v, output, num_decode, num_toks):
        if self.layer_idx == 0:
            kv_page_indices_np, kv_page_indptr_np, kv_last_page_len_np  = page_table.page_table(self.req_ids[self.num_prefill_tok:])  
            kv_page_indices = torch.from_numpy(kv_page_indices_np).cuda()
            kv_page_indptr = torch.from_numpy(kv_page_indptr_np).cuda()
            kv_last_page_len = torch.from_numpy(kv_last_page_len_np).cuda()
            print(f"decode: kv_page_indices= {kv_page_indices}, kv_page_indptr {kv_page_indptr}, kv_last_page_len {kv_last_page_len} ")
            
            
            PyPlugin.decode_page_table = kv_page_indices, kv_page_indptr, kv_last_page_len, kv_page_indices_np, kv_page_indptr_np, kv_last_page_len_np
        else:
            kv_page_indices, kv_page_indptr, kv_last_page_len = PyPlugin.decode_page_table
            
        paged_kv_cache = global_kv[self.layer_idx]
        paged_kv_cache[0][999][kv_last_page_len-1] = k
        paged_kv_cache[1][999][kv_last_page_len-1] = v
        
        self.decode_wrapper.plan(
                    kv_page_indptr,
                    kv_page_indices,
                    kv_last_page_len,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.page_size,
                    pos_encoding_mode="ROPE_LLAMA",
                    data_type=torch.float16
                )
        print(f"xxxx query_states", q.shape)
        # query_states = query_states[0]
        
        attn_output = self.decode_wrapper.run(q, paged_kv_cache, out = output)
        print(" decode output= ",output, output.shape)

    def prefill_forward(self, q, k, v, output, num_prefill_tok, num_toks):
        num_toks_gpu = torch.tensor(num_toks, device='cuda')
        cumulative_sum_gpu = torch.cumsum(num_toks_gpu, dim=0)
        zero = torch.tensor([0], device='cuda', dtype=cumulative_sum_gpu.dtype)
        q_indptr = torch.cat((zero, cumulative_sum_gpu), dim=0)

        kv_indptr = q_indptr
        
        self.prefill_wrapper.plan(
                    q_indptr,
                    kv_indptr,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    causal=True, pos_encoding_mode='ROPE_LLAMA',
                )
        print(f'prefill output={output.shape}', q.shape, v.shape)

        attn_output = self.prefill_wrapper.run(q, k,
                                v,
                                out=output)
        self.prefill_upadate_kvcache(k, v, self.req_ids[:num_prefill_tok], kv_indptr, num_toks_gpu)
        
    def prefill_upadate_kvcache(self, k, v, req_ids, kv_indptr, num_toks_gpu):
        paged_kv_cache = global_kv[self.layer_idx]
        
        if self.layer_idx == 0:
            kv_page_indices, kv_page_indptr, kv_last_page_len  = page_table.page_table(req_ids)  
            print(f"kv_page_indices= {kv_page_indices}, kv_page_indptr {kv_page_indptr}, kv_last_page_len {kv_last_page_len} ")
            kv_page_indices = torch.from_numpy(kv_page_indices).cuda()
            kv_page_indptr = torch.from_numpy(kv_page_indptr).cuda()
            kv_last_page_len = torch.from_numpy(kv_last_page_len).cuda()
            
            seq = num_toks_gpu# flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, 16)
            
            batch_indices, positions = flashinfer.get_batch_indices_positions(kv_indptr, seq, k.shape[0])
            print(f'seq={seq},positions={positions},batch_indices={batch_indices},k.shape[0]={k.shape[0]}')
            
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
        print(kv_page_indptr)
        print("paged_kv_cache in prefill: ", paged_kv_cache[0][999,4,2,1], k[4,2,1])
        assert paged_kv_cache[0][999,4,2,1] == k[4,2,1]
      
        
if __name__ == '__main__':
    import time
    time.sleep(10)

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
    #     id_type req_id;
    # int32_t req_tokens{0};
    # int32_t new_tokens{0};
    # int32_t max_new_tokens{0};
    # int32_t max_tokens{0};
    io[hami.TASK_REQUEST_ID_KEY] = "id"
    io[hami.TASK_MSG_KEY] = hami.TypedDict({"req_tokens": 5,
                                            "max_new_tokens": 7,
                                            "max_tokens":4096})
    model(io)
    # print([x.shape for x in io['result']])
    # print(io['result'])
    
    tokenizer = AutoTokenizer.from_pretrained('exported_params/')
    q = hami.default_queue("net_out")
    while not q.empty():
        data = q.get()
        print(data['data'])
        
        text = tokenizer.decode(data['data'], skip_special_tokens=True)
        print(text)