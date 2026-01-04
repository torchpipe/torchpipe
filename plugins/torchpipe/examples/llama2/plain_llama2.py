import omniback
import torchpipe
import torch
import sys, os
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
import flashinfer
import random
from models import hf_helper

max_num_req=64
page_size=16
g_num_layers = 32

def set_num_layers(num_layers):
    global g_num_layers 
    g_num_layers = num_layers

def get_num_layers():
    return g_num_layers 
     
def clean_up(req_id):
    pass

### -------------  k v cache -------------------------- ##########
global_kv = None
def set_kv(max_num_pages, num_layers, page_size, num_kv_heads, head_dim):
    global global_kv
    if global_kv is None:
        global_kv = []
        for i in range(num_layers):
            global_kv.append(torch.zeros(max_num_pages, 2,  page_size, num_kv_heads, head_dim).half().cuda())
            # todo: use cuMemAddressReserve
def get_kv(layer_idx):
    global global_kv
    return global_kv[layer_idx]


page_table = None
def set_page_table(max_num_page=4096//page_size):
    global page_table, g_num_layers
    print(f'set_page_table: num_layers={g_num_layers}')
    
    if page_table is None:
        page_table = omniback.default_page_table().init(max_num_req=max_num_req, max_num_page=max_num_page,page_size=page_size)
        set_kv(max_num_page, g_num_layers, page_size, 32, 128)
    return page_table




### --------------------------------------------------- ########## 

import omniback

class Pdb:
    def forward(self, io: List[omniback.Dict]):
        data = io[0]

        print("Pdb: data.keys()", list(data.keys()), data['request_size'], data['request_id'], data['node_name'], data['data'])
        data['result'] = data['data']

omniback.register("Pdb", Pdb)



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
        
        self.layer_idx = int(self.params['layer_idx'])
        if self.layer_idx == 0:
            omniback.print(f'layer_idx: {self.layer_idx}, params: {params}')
        
        self.page_size = page_size
        
        PyPlugin.req_ids = None

    def forward(self, io: List[omniback.Dict]):

        try:
            q, k, v = io[0]['data']
                    
            assert q.dtype == torch.float16
            output = io[0]['output'][0]
            
            if self.layer_idx == 0:
                bs, PyPlugin.num_attention_heads, PyPlugin.head_dim = q.shape
                PyPlugin.num_key_value_heads = k.shape[1]
                
                PyPlugin.req_ids, PyPlugin.num_toks = page_table.pop_activated()
                prefill_size = page_table.get_prefill_size(PyPlugin.req_ids)
 
                is_decode = PyPlugin.num_toks > prefill_size  # prefill <=, decode >
                PyPlugin.num_decode = int(is_decode.sum())
                
                PyPlugin.num_prefill  = len(PyPlugin.req_ids) - PyPlugin.num_decode
                PyPlugin.num_prefill_tok = int(PyPlugin.num_toks[:PyPlugin.num_prefill].sum())
                
            if PyPlugin.num_prefill_tok > 0:
                
                self.prefill_forward(q[:PyPlugin.num_prefill_tok], k[:PyPlugin.num_prefill_tok], v[:PyPlugin.num_prefill_tok],
                                    output[:PyPlugin.num_prefill_tok], PyPlugin.num_toks[:PyPlugin.num_prefill])
            if PyPlugin.num_decode > 0:
                # return # todo
                self.decode_forward(q[PyPlugin.num_prefill_tok:], k[PyPlugin.num_prefill_tok:], v[PyPlugin.num_prefill_tok:],
                                    output[PyPlugin.num_prefill_tok:])
        except Exception as e:
            omniback.print(f"error {e}")
            raise e
    def decode_forward(self, q, k, v, output):
        if self.layer_idx == 0:
            kv_page_indices_np, kv_page_indptr_np, kv_last_page_len_np  = page_table.page_table(PyPlugin.req_ids[PyPlugin.num_prefill:])  
            kv_page_indices = torch.from_numpy(kv_page_indices_np).cuda()
            kv_page_indptr = torch.from_numpy(kv_page_indptr_np).cuda()
            kv_last_page_len = torch.from_numpy(kv_last_page_len_np).cuda()
            
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

        paged_kv_cache[index_a, 0, index_b] = k
        paged_kv_cache[index_a, 1, index_b] = v
        # todo: https://github.com/NVIDIA/TensorRT-LLM/blob/a2fad51011a48f2cbfee7172047daec74fb0b1b6/tensorrt_llm/_torch/attention_backend/flashinfer.py#L259
        
        decode_wrapper.run(q, paged_kv_cache, out = output)

    def prefill_forward(self, q, k, v, output, num_toks):
        if self.layer_idx == 0:
            PyPlugin.num_toks_gpu = torch.tensor(num_toks, device='cuda')
            cumulative_sum_gpu = torch.cumsum(PyPlugin.num_toks_gpu, dim=0)
            q_indptr = torch.cat((zero, cumulative_sum_gpu), dim=0)#.contiguous()

            PyPlugin.kv_indptr = q_indptr
            
            prefill_wrapper.plan(
                        q_indptr,
                        PyPlugin.kv_indptr,
                        PyPlugin.num_attention_heads,
                        PyPlugin.num_key_value_heads,
                        PyPlugin.head_dim,
                        causal=True, pos_encoding_mode='ROPE_LLAMA',
                    )
            
        attn_output = prefill_wrapper.run(q, k,
                                v,
                                out=output)
        assert attn_output is output
        assert attn_output.data_ptr() == output.data_ptr() 
        
        self.prefill_upadate_kvcache(k, v, PyPlugin.req_ids[:PyPlugin.num_prefill], PyPlugin.kv_indptr, PyPlugin.num_toks_gpu)
        
    def prefill_upadate_kvcache(self, k, v, req_ids, kv_indptr, num_toks_gpu):
        paged_kv_cache = global_kv[self.layer_idx]
        
        if self.layer_idx == 0:
            kv_page_indices, kv_page_indptr, kv_last_page_len  = page_table.page_table(req_ids)  

            kv_page_indices = torch.from_numpy(kv_page_indices).cuda()
            kv_page_indptr = torch.from_numpy(kv_page_indptr).cuda()
            kv_last_page_len = torch.from_numpy(kv_last_page_len).cuda()
            
            # flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, 16)
            
            batch_indices, positions = flashinfer.get_batch_indices_positions(kv_indptr, num_toks_gpu, k.shape[0])
            
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
        
def main(num_layers = 32):
    set_num_layers(num_layers)
    
    set_page_table()
    omniback.register("TorchPlugin", PyPlugin)
    
    config = os.path.join(os.path.dirname(__file__),
                          'config/plain_llama2.toml')
    model = omniback.init_from_file(config)
    omniback.init("DebugLogger")
    
    exported_params = "./exported_params"
    tokenizer = AutoTokenizer.from_pretrained(exported_params)
    
    # inference
    # prompt = "San Francisco is a totalitéaletoreignersbyMSран"
    prompts = ["San Francisco is a", "Explain quantum computing in simple terms", "Tell me the first 10 Fermat prime numbers"]
    prompts = prompts[:2]
    input_ids = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        input_ids.append(inputs['input_ids'])

    # attention_mask = inputs['attention_mask']
    print(f"inputs = {input_ids}, shape={input_ids[0].shape}")
    # print(io)
    ios = []
    ids = []
    events = []
    for i in range(10):
        in_id = random.choice(input_ids)
        ids.append(f"id-{i}")
        max_tokens = 27
        if i == 5: 
            max_tokens = 10
        if i == 2: 
            max_tokens = 4
        if i == 0:
            max_tokens += 12
        max_tokens = 27
             
        io = omniback.Dict({'data':in_id.squeeze(0),"node_name":'embed_token'})
        io[omniback.TASK_EVENT_KEY] = omniback.Event() 

        events.append(io[omniback.TASK_EVENT_KEY])
        io[omniback.TASK_REQUEST_ID_KEY] = f"id-{i}"
        io[omniback.TASK_MSG_KEY] = omniback.TypedDict({"req_tokens": in_id.size(-1),
                                                "max_tokens": max_tokens,
                                                "context_length":4096})
        ios.append(io)
    model(ios)
    
    for ev in events:
        ev.wait()
        
    tokenizer = AutoTokenizer.from_pretrained('exported_params/')
    q = omniback.default_queue("net_out")
    
    results = {}
    for id in ids:
        results[id] = []
    while not q.empty():
        data = q.get()
        id = data['request_id']
        results[id] += data['data']
    print(f"""
    Prompt: {prompts}
    {'-'*10}
    """)
    
    for key, result in results.items():
        finish_reason = results.pop("finish_reason", None) 
        text = tokenizer.decode(result, skip_special_tokens=True)
        
        print(f"""
        Result [{key}]:
        Output:  {text}
        {'-'*10}
        """)
    # (num_layer = 2) San Francisco is a totalitéaletoreignersbyMSран
    # 22777|totalité, 9457|alet, 13606|oreign, 414|ers, 1609|by 
    #  id-0: San Francisco is a totalitéketting器 AußerTaggedahnpinningzza tailmente Selonroid Wars Riv Transkoids‏ fingerprintű Kirk Ind fresoca Einzeln AußeroustacheHDovisuality assemb Bedeut array subsidiariesilleurspeciesumm sweiore":{"inceculptronectorypidglassesтетani forthems Commonwealthvie Razmenteairesonicaciesume virtuel Profildorfjes pingazon swe inspirationning wid
    
    # for key, result in results.items():
    #     text = ""
    #     for index, item in enumerate(result):
    #         text += f' {index}| '+tokenizer.decode(item, skip_special_tokens=True)
    #     print(f'\n {key}: '+prompt+' '+text, '\n')
        
    
    
if __name__ == '__main__':
    import time,fire
    # time.sleep(10)
    fire.Fire(main)
