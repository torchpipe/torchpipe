import omniback
import torchpipe
import torch
import sys, os
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
import flashinfer
import random
from models.cos_sin_attention_mask import  generate_cos_sin_attention_mask

max_num_req=64
page_size=16
g_num_layers = 2

def set_num_layers(num_layers):
    global g_num_layers
    g_num_layers = num_layers

def get_num_layers():
    return g_num_layers

### -------------  k v cache -------------------------- ##########
# global_kv = None
# def set_kv(max_num_pages, num_layers, page_size, num_kv_heads, head_dim):
#     global global_kv
#     if global_kv is None:
#         global_kv = []
#         for i in range(num_layers):
#             global_kv.append(torch.zeros(max_num_pages, 2,  page_size, num_kv_heads, head_dim).half().cuda())
#             # todo: use cuMemAddressReserve
# def get_kv(layer_idx):
#     global global_kv
#     return global_kv[layer_idx]


page_table = None
def set_page_table(max_num_page=4096//page_size):
    global page_table, g_num_layers
    print(f'set_page_table: num_layers={g_num_layers}')
    
    if page_table is None:
        page_table = omniback.default_page_table().init(max_num_req=max_num_req, max_num_page=max_num_page,page_size=page_size)
        # set_kv(max_num_page, g_num_layers, page_size, 32, 128)
    return page_table


from dataclasses import dataclass, astuple, field

@dataclass
class RequestStatus:
    kvcache: Tuple[torch.Tensor]
    cos: torch.Tensor
    sin : torch.Tensor
    att_mask: torch.Tensor
    
request_status = {}

def clean_up(req_id):
    request_status.pop(req_id, None)
    # del request_status[req_id]
### --------------------------------------------------- ########## 

import omniback

class Pdb:
    def forward(self, io: List[omniback.Dict]):
        data = io[0]

        print("Pdb: data.keys()", list(data.keys()), data['request_size'], data['request_id'], data['node_name'], data['data'])
        data['result'] = data['data']

omniback.register("Pdb", Pdb)


attention_kernel = omniback.init_from_file('config/attention_kernel_streams.toml')
prefill_attn = omniback.get('prefill.0')
decode_attns = [omniback.get(f'decode{i}.0') for i in range(2)]
# decode_attn = omniback.init("Pool(decode0.0, decode1.0)")
# https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-semantics
streams = [torch.cuda.Stream() for i in range(2)]
def decode_attn(ios, index):
    index = index%len(streams)
    if index == 0:
        decode_attns[0](ios)
    else:
        s = streams[index]
        s.wait_stream(torch.cuda.current_stream())  # NEW!
        with torch.cuda.stream(s):
            decode_attns[index](ios)
        torch.cuda.current_stream().wait_stream(s)
    
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
        # if True:
            q, k, v = io[0]['data']
                    
            assert q.dtype == torch.float16
            output = io[0]['output'][0]
            
            if self.layer_idx == 0:
                bs, PyPlugin.num_attention_heads, PyPlugin.head_dim = q.shape
                PyPlugin.num_key_value_heads = k.shape[1]
                
                PyPlugin.req_ids, PyPlugin.num_toks = page_table.pop_activated()
                
                prefill_size = page_table.get_prefill_size(PyPlugin.req_ids)
                
                PyPlugin.is_prefill = PyPlugin.num_toks == prefill_size  # prefill <=, decode >
                PyPlugin.num_prefill = int(PyPlugin.is_prefill.sum())
                
                adjusted_sizes = np.where(PyPlugin.is_prefill, prefill_size, 1)

                PyPlugin.cumsum = torch.cumsum(torch.from_numpy(adjusted_sizes), 0)
                PyPlugin.split_indices = PyPlugin.cumsum[:-1]
                # omniback.print(f"current_size={adjusted_sizes}, split_indices={PyPlugin.split_indices} {PyPlugin.is_prefill}")
                
                
                PyPlugin.num_decode  = len(PyPlugin.req_ids) - PyPlugin.num_prefill
                PyPlugin.num_prefill_tok = int(PyPlugin.num_toks[:PyPlugin.num_prefill].sum())
                
                for id, is_prefill, num_tok in zip(PyPlugin.req_ids, PyPlugin.is_prefill, PyPlugin.num_toks):
                    if is_prefill:
                        cos, sin, att_mask = generate_cos_sin_attention_mask(num_tok, num_tok, is_prefill)
                        request_status[id] = RequestStatus(kvcache=[None]*g_num_layers, cos=cos, sin=sin, att_mask=att_mask)
                    else:
                        cos, sin, att_mask = generate_cos_sin_attention_mask(1, num_tok, is_prefill)    
                        status = request_status[id]
                        status.cos = cos    
                        status.sin = sin
                        status.att_mask = att_mask
            
            q_split = torch.tensor_split(q, PyPlugin.split_indices)
            k_split = torch.tensor_split(k, PyPlugin.split_indices)
            v_split = torch.tensor_split(v, PyPlugin.split_indices)
            o_split = torch.tensor_split(output, PyPlugin.split_indices)
            
            index = 0
            for id, qs, ks, vs, os, is_prefill, num_tok in zip(PyPlugin.req_ids, q_split, k_split, v_split, o_split, PyPlugin.is_prefill, PyPlugin.num_toks):
                index += 1
                status = request_status[id]
                if is_prefill:
                    #  omniback.print(f"prefill: {id}, {self.layer_idx}, s={qs.shape}, status.cos, status.sin, status.att_mask shape = {status.cos.shape}, {status.sin.shape}, {status.att_mask.shape}")
                    shape = qs.shape
                    inputs = [qs.view(1, shape[0], shape[1]*shape[2]), ks.view(1, shape[0], shape[1]*shape[2]), vs.view(1, shape[0], shape[1]*shape[2]), status.cos, status.sin, status.att_mask]
                    
                    dtype = qs.dtype
                    device = qs.device

                    new_shape = (1, 32, shape[0], 128)

                    out_k = torch.empty(new_shape, dtype=dtype, device=device)
                    out_v = torch.empty(new_shape, dtype=dtype, device=device)
                    # print(f'out_k = {out_k.shape}/{out_v.shape}')
                    # torch.cuda.synchronize()

                    ios = omniback.Dict({'data':inputs,"node_name": 'prefill', "output":[os.view(1, shape[0], shape[1]*shape[2]), out_k, out_v]}) # , out_k, out_v
                    # assert False, "TODO: pre-defined outputs for k v cache"
                    prefill_attn(ios)
                    # k2 = ios['result'][2]
                    # omniback.print(f'out_v={out_v.shape}, k2={k2.shape}, {out_v.data_ptr() == k2.data_ptr()}')
                    # out_k, out_v = ios['result'][1], ios['result'][2]
                    status.kvcache[self.layer_idx] = (out_k, out_v)
                    # import pdb; pdb.set_trace()
                else:
                    k, v = status.kvcache[self.layer_idx]
                    status.kvcache[self.layer_idx] = None
                    # omniback.print(f"decode: {id}, q={qs.shape}, {self.layer_idx}, status.cos, status.sin, status.att_mask shape = {status.cos.shape}, {status.sin.shape}, {status.att_mask.shape}, num_toks={PyPlugin.num_toks}")
                    shape = qs.shape
                    
                    dtype = qs.dtype
                    device = qs.device

                    new_shape = (1, 32, k.shape[-2]+1, 128)

                    out_k = torch.empty(new_shape, dtype=dtype, device=device)
                    out_v = torch.empty(new_shape, dtype=dtype, device=device)
                    
                    inputs = [qs.view(1, shape[0], shape[1]*shape[2]), ks.view(1, shape[0], shape[1]*shape[2]), vs.view(1, shape[0], shape[1]*shape[2]), status.cos, status.sin, status.att_mask, k, v]
                    ios = omniback.Dict({'data':inputs,"node_name":'decode', "output":[os.view(1, shape[0], shape[1]*shape[2]), out_k, out_v]}) #, out_k, out_v
                    decode_attn(ios, index)
                    # status.kvcache[self.layer_idx] = (ios['result'][1], ios['result'][2])
                    status.kvcache[self.layer_idx] = (out_k, out_v)
                    # import pdb; pdb.set_trace()
            
            assert q_split[0].data_ptr() == q.data_ptr()
            
        except Exception as e:
            omniback.print(f"error {e}")
            raise e
        
def main(num_layers = 2):
    set_num_layers(num_layers)
    
    set_page_table()
    omniback.register("TorchPlugin", PyPlugin)
    
    model = omniback.init_from_file('config/plain_llama2.toml')
    omniback.init("DebugLogger")
    
    exported_params = "./exported_params"
    tokenizer = AutoTokenizer.from_pretrained(exported_params)
    
    # inference
    # prompt = "San Francisco is a totalitéaletoreignersbyMSран"
    prompts = ["San Francisco is a", "Explain quantum computing in simple terms", "Tell me the first 10 Fermat prime numbers"]
    prompts = [prompts[0], prompts[1]]
    # prompts = [prompts[0]]
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
