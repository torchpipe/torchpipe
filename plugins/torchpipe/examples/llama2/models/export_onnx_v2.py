import os, types
from typing import Optional, Tuple
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

import os, tempfile, shutil
import subprocess
import sys
import importlib
import fire

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
# from models.partial_hf import get_hf_model
from models import hf_helper

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

 
print(transformers.__version__)



class TorchPlugin(torch.nn.Module):
    # export plugin as local function need PEP 526
    params: str

    def __init__(self):
        super().__init__() 
        self.params = ""

    def add_params(self, key, value):
        if len(self.params) != 0:
            self.params += ";"
        self.params += f"{key}={value}"
        
    def forward(self, q, k, v):
        return q + k + v

def _modify_attention(attention):
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

        query_states = self.q_proj(hidden_states).view(-1, self.config.num_attention_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(-1, self.config.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(-1, self.config.num_key_value_heads, self.head_dim)
  
        attn_output = self.batchless_attn(query_states, key_states, value_states).view(-1,  self.config.num_attention_heads* self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None
    attention.forward = types.MethodType(forward, attention)
    tp = TorchPlugin()
    tp.add_params("layer_idx", attention.layer_idx)
    tp.add_params("num_output", 1)
    tp.add_params("num_input", 3)
    tp.add_params("dtype", 'fp16') # or fp16,fp16,fp16,fp16
    # tp.add_params("workspace", 1024*1024*128)
    attention.batchless_attn = tp

def _modify_decode_layers(layers):
    for layer in layers:
        # print(f'_modify_decode_layers: {layer}')
        _modify_attention(layer.self_attn)

def _modify_llama_model(llama_model):
    def llama_forward(
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
        
        for index, decoder_layer in enumerate(self.layers):
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

    llama_model.forward = types.MethodType(llama_forward, llama_model)
    
def save_embed_tokens(model, out_dir = 'model_files/'):
    with torch.no_grad():
        data = model.model.embed_tokens.weight.requires_grad_(False).cpu()
        torch.save(os.path.join(out_dir, 'embed_tokens.pt'), data)
        print(model.model.embed_tokens.weight.shape)
    
    
class Wrapper(torch.nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    def forward(self, inputs_embeds):
        with torch.no_grad():
            out = self.llm.forward(input_ids = None,
                            attention_mask = None,
                            position_ids = None,
                            past_key_values = None,
                            inputs_embeds = inputs_embeds)
        
        print('out', out)
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

        if index_select is not None:
            assert index_select.shape[0] <= hidden_states.shape[0]

            hidden_states = hidden_states.index_select(0, index_select)

        logits = self.lm_head(hidden_states)
        
        return (logits, )
    llm.forward = types.MethodType(forward, llm)

            
def export_batchable(model,  out_dir = 'model_files/', use_index_select = True):

    llama_model = model.model
    _modify_decode_layers(llama_model.layers) 
    num_layers = len(llama_model.layers)
    # need_sim = num_layers < 16
    need_sim = True
    _modify_llama_model(llama_model)
    
    _modify_causal_llm(model)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "batchable.onnx")
    if need_sim:
        out_tmp_dir = os.path.join(out_dir, ".temp/")
        os.makedirs(out_tmp_dir, exist_ok = True)
        out_tmp_path = os.path.join(out_tmp_dir, "batchable.onnx")
    else:
        out_tmp_path = out_path

    def _export_batchable(llm):
        if use_index_select:
            dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'index_select': {0: 'request_size'}}
            index_select = torch.zeros((2,)).to(torch.long).to(llm.model.device)
            index_select[-2] = 2939
            index_select[-1] = 2940
            print(index_select, llm.model.device)
            input_names = ['inputs_embeds', 'index_select']
        else:
            dynamic_axes = {'inputs_embeds': {0: 'seq_len'}, 'logits': {0: 'seq_len'}}
            input_names = ['inputs_embeds']
        
        inputs_embeds = torch.zeros((2941, llm.model.embed_tokens.weight.shape[-1])).to(llm.model.device, torch.float16)
        print(f'start exporting {out_path}')
        torch.onnx.export(hf_helper.EmbedsAsInputWrapper(model),
                        args=(inputs_embeds, index_select),
                        f=out_tmp_path,
                        opset_version=17,
                        input_names=input_names,
                        output_names=['logits'],
                        dynamic_axes=dynamic_axes,
                        export_modules_as_functions={TorchPlugin})
        
                        # custom_opsets={"CustomTorchOps": 1}

        # raise RuntimeError(out_tmp_path)
        if need_sim:
            assert 0 == subprocess.call(
                ["onnxslim", out_tmp_path, out_path, '--save-as-external-data'])

            shutil.rmtree(out_tmp_dir)
        
        print(f'Batchable: {out_path} saved.')
        # def repair_prefix(model_path):
        #     import onnx
        #     model = onnx.load(model_path)
        #     for node in model.graph.node:
                
        #         # 将 "/model/layers.0/linear/..." 转为 "linear_..."
        #         # new_name = node.name.split("/")[-1]  # 取最后一段名称
        #         if "TorchPlugin" in  node.name:
        #             new_name = node.name.split("/")[-1].replace(".", "_")
        #             print(node.name, f" replaced by {new_name} node.op = {node.op_type}")
        #             node.name = new_name.replace(".", "_")
        #         else:
        #             print(f"node.name: = {node.name:}")
                    
        #     onnx.save(model, model_path)
        # repair_prefix(out_path)
                

    
    
    _export_batchable(model)
    # exit(0)

def export_batchless_prefill(model, out_dir = 'model_files/'):
    llama_model = model.model
    
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    
    def batchless_forward(self, query_states, key_states, value_states, cos, sin, attention_mask):
        bsz = 1
        assert query_states.shape[0] == bsz
        q_len = query_states.shape[-2]

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
        past_key, past_value = key_states, value_states
        # past_key = key_states.transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, self.config.num_attention_heads* self.head_dim)
        return attn_output, past_key, past_value

    origin_forward = llama_model.layers[0].self_attn.forward
    llama_model.layers[0].self_attn.forward = types.MethodType(batchless_forward, llama_model.layers[0].self_attn)
    
    bsz = 1
    q_len = 244

    hidden_size = llama_model.embed_tokens.weight.shape[-1]
    device = llama_model.device
    query_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    key_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    value_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    cos = torch.zeros((bsz, q_len, hidden_size//32), dtype=torch.float16, device=device)
    sin = torch.zeros((bsz, q_len, hidden_size//32), dtype=torch.float16, device=device)
    attention_mask = torch.zeros((1,1, q_len, q_len), dtype=torch.float16, device=device)

    dynamic_axes={'query_states': {1: 'seq_len'}, 'key_states': {1: 'seq_len'},
                    'value_states': {1: 'seq_len'}, 
                    'attn_output': {1: 'seq_len'}, 'cos': {1: 'seq_len'},
                    'sin': {1: 'seq_len'},'attention_mask': {2: 'seq_len',3: 'seq_len'}}
    dynamic_axes.update({'past_key_output':{2: 'seq_len'}, 'past_value_output':{2: 'seq_len'}})
    out_path = os.path.join(out_dir,"./batchless_prefill.onnx")

    torch.onnx.export(llama_model.layers[0].self_attn,
                    args=(query_states,key_states, value_states, cos, sin, attention_mask),
                    f=out_path,
                    opset_version=17,
                    input_names=['query_states','key_states', 'value_states','cos','sin', 'attention_mask'],
                    output_names=['attn_output', 'past_key_output', 'past_value_output'] ,
                    dynamic_axes=dynamic_axes)
    print(f'Prefill Batchless CausalLM: {out_path} saved.')
    try:
        import onnxsim
    except ImportError:
        print("onnxsim not found, installing...")
        install_package("onnxsim")
    
    os.system(f"onnxsim {out_path} {out_path}")
    
    llama_model.layers[0].self_attn.forward = origin_forward


def export_batchless_decode(llm, out_dir = 'model_files/'):
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
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        assert past_key is not None
        key_states = torch.cat([past_key, key_states], dim=2) 
        value_states = torch.cat([past_value, value_states], dim=2) 
            
        past_key, past_value = key_states, value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None  # no matter the length, we just slice it

        causal_mask = attention_mask

        attn_weights = attn_weights + causal_mask
            
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
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
    num_attention_heads = model.layers[0].self_attn.config.num_attention_heads
    num_key_value_heads = model.layers[0].self_attn.config.num_key_value_heads
    num_key_value_groups = model.layers[0].self_attn.num_key_value_groups
 
    print("hidden_size/num_attention_heads ", hidden_size, num_attention_heads)
    print(f"num_key_value_groups={num_key_value_groups}")
    device = model.device
    query_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    key_states = torch.zeros((bsz, q_len, hidden_size//num_key_value_groups), dtype=torch.float16, device=device)
    value_states = torch.zeros((bsz, q_len, hidden_size//num_key_value_groups), dtype=torch.float16, device=device)
    
    past_key = torch.zeros((1,num_key_value_heads//num_key_value_groups, past, hidden_size//num_key_value_heads), dtype=torch.float16, device=device)
    past_value = torch.zeros((1,num_key_value_heads//num_key_value_groups, past, hidden_size//num_key_value_heads), dtype=torch.float16, device=device)
    
    cos = torch.zeros((bsz, q_len, hidden_size//num_attention_heads), dtype=torch.float16, device=device)
    sin = torch.zeros((bsz, q_len, hidden_size//num_attention_heads), dtype=torch.float16, device=device)
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
    
    
def save_embed_tokens(model, out_dir = 'model_files/'):
    save_name = os.path.join(out_dir, 'embed_tokens.pt')
    tensor = model.model.embed_tokens.weight.requires_grad_(False).data.cpu()
    torch.save(tensor, save_name)
    print(f"Embedding tokens saved to {save_name}")

def export_all(model_id='meta-llama/Llama-2-7b-chat-hf', num_layers=2, out_dir = 'exported_params/'):
    model, tokenizer, num_layers = hf_helper.get_hf_model(model_id, num_layers=num_layers, use_flashinfer=False)
    prompt = "San Francisco is a"
    prompt = "Tell me the first 10 Fermat prime numbers"
    result = hf_helper.generate_text(model, tokenizer, prompt, 37)
    print(f"\nnum_layers = {num_layers}, Generated text: {result}")
    #exit(0)
    # num_layers = 2, Generated text: San Francisco is a totalitéaletoreignersbyMSран /or/ totalitéketting器 AußerTagged
    # num_layers = 32, Generated text: San Francisco is a city in Northern California that is known
    export_batchable(model, out_dir = out_dir, use_index_select = True)
    export_batchless_prefill(model, out_dir=out_dir)
    export_batchless_decode(model, out_dir=out_dir)
    save_embed_tokens(model, out_dir=out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == '__main__':
    fire.Fire(export_all)
    
    # batch-compatible / batch-incompatible
    # batchable / non-batchable
    # batch-supporting / batch-unsupported
    # Batch-Aware / Batch-Unaware
