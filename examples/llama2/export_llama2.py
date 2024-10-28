import os, types
from typing import Optional, Tuple
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
print(transformers.__version__)
assert transformers.__version__ == '4.44.2'
# logger = logging.getLogger(__name__)

# # 将模型移动到 GPU
# model.to(device)

# 准备输入文本

# print(model)

# 使用 fp16 进行推理

def _modify_attention(attention):
    
    
    class DummyTorchPluginOp(torch.autograd.Function):
        @staticmethod
        def symbolic(g, q, k, v):
            args = [q, k, v]
            # These become the operator attributes.
            kwargs = {}
            from torch.onnx.symbolic_helper import _get_tensor_sizes
            output_type = q.type().with_sizes(_get_tensor_sizes(q))
            return g.op("CustomTorchOps::TorchPlugin", *args,
                        **kwargs).setType(output_type)

        @staticmethod
        def forward(ctx, q,k,v):
            return q
            
    class DummyTorchPlugin(torch.nn.Module):
        def forward(self, q,k,v):
            x = DummyTorchPluginOp.apply(q,k,v)
            return x


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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
  
        attn_output = self.batchless_attn(query_states, key_states, value_states)
        attn_output = self.o_proj(attn_output)
        
 
        return attn_output, None, None
    attention.forward = types.MethodType(forward, attention)
    attention.batchless_attn = DummyTorchPlugin()


def _modify_decode_layers(layers):
    for layer in layers:
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

        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        
        for index, decoder_layer in enumerate(self.layers):
            print(f'decoder_layer {index}')
            if True:
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
    # llama_model.forward = types.MethodType(forward_batchful, llama_model)
def save_embed_tokens(model, save_dir = 'model_files/'):
    with torch.no_grad():
        data = model.model.embed_tokens.weight.requires_grad_(False).cpu()
        torch.save(os.path.join(save_dir, 'embed_tokens.pt'), data)
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

def _modify_causal_llm(llm, return_last_seq_len = False):
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
        if return_last_seq_len:
            hidden_states = outputs[0][-1,:].unsqueeze(0)
        else:
            hidden_states = outputs[0]
        # print(f'self.lm_head = {self.lm_head.weight.shape}')
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        
        return (logits,)
    llm.forward = types.MethodType(forward, llm)
    # llm.lm_head = torch.nn.Linear(llm.model.embed_tokens.weight.shape[-1], llm.model.config.vocab_size, bias=False)
    # new_lm_head = torch.nn.Linear(llm.lm_head.weight.shape[1], 1, bias=False)
    # assert llm.lm_head.bias is None
    # new_lm_head.weight.data = llm.lm_head.weight.data[-1, :].unsqueeze(0)
    # llm.lm_head = new_lm_head

            
def save_batchful(model, num_layers = None, save_path = 'model_files/', return_last_seq_len = False):
    llama_model = model.model
    _modify_decode_layers(llama_model.layers) 
    _modify_llama_model(llama_model)
    
    _modify_causal_llm(model, return_last_seq_len)
    
    if num_layers is not None:
        llama_model.layers = llama_model.layers[:num_layers]
    
    
    def export_batchful(llm, save_path = save_path, return_last_seq_len = False):
        model = Wrapper(llm).eval()

        # assert past_key_values is not None
        if return_last_seq_len:
            dynamic_axes={'inputs_embeds': {0: 'seq_len'}}
        else:
            dynamic_axes={'inputs_embeds': {0: 'seq_len'}, 'logits': {0: 'seq_len'}}
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, "./batchful.onnx")

        inputs_embeds = torch.zeros((244, llm.model.embed_tokens.weight.shape[-1])).to(model.llm.device, torch.float16)
        print(f'start exporting {out_path}')
        torch.onnx.export(model,
                        args=(inputs_embeds,),
                        f=out_path,
                        opset_version=17,
                        input_names=['inputs_embeds'],
                        output_names=['logits'] ,
                        dynamic_axes=dynamic_axes)
        print(f'Batchful CausalLM: {out_path} saved.')
    
    export_batchful(model, save_path, return_last_seq_len=return_last_seq_len)
    exit(0)

def save_prefill_batchless(model, save_path = 'model_files/'):
    llama_model = model.model
    
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    

    def batchless_forward(self, query_states, key_states, value_states, cos, sin, attention_mask, past_key = None, past_value = None):
        bsz = 1
        assert query_states.shape[0] == bsz
        q_len = query_states.shape[-2]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2) 
            value_states = torch.cat([past_value, key_states], dim=2) 
            
        past_key, past_value = key_states, value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            causal_mask = attention_mask
            # print(causal_mask, attention_mask.shape)
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)
        return attn_output, past_key, past_value

    origin_forward = llama_model.layers[0].self_attn.forward
    llama_model.layers[0].self_attn.forward = types.MethodType(batchless_forward, llama_model.layers[0].self_attn)
    
    bsz = 1
    q_len = 244
    past_key_values_length = 0
    hidden_size = llama_model.embed_tokens.weight.shape[-1]
    device = llama_model.device
    query_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    key_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    value_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    past_key, past_value = None, None
    cos = torch.zeros((bsz, q_len, hidden_size//32), dtype=torch.float16, device=device)
    sin = torch.zeros((bsz, q_len, hidden_size//32), dtype=torch.float16, device=device)
    attention_mask = torch.zeros((1,1, q_len, q_len), dtype=torch.float16, device=device)

    dynamic_axes={'query_states': {1: 'seq_len'}, 'key_states': {1: 'seq_len'},
                    'value_states': {1: 'seq_len'}, 
                    'attn_output': {1: 'seq_len'}, 'cos': {1: 'seq_len'},
                    'sin': {1: 'seq_len'},'attention_mask': {2: 'seq_len',3: 'seq_len'}}
    dynamic_axes.update({'past_key_output':{2: 'seq_len'}, 'past_value_output':{2: 'seq_len'}})
    out_path = os.path.join(save_path,"./batchless_prefill.onnx")

    torch.onnx.export(llama_model.layers[0].self_attn,
                    args=(query_states,key_states, value_states, cos, sin, attention_mask),
                    f=out_path,
                    opset_version=17,
                    input_names=['query_states','key_states', 'value_states','cos','sin', 'attention_mask'],
                    output_names=['attn_output', 'past_key_output', 'past_value_output'] ,
                    dynamic_axes=dynamic_axes)
    print(f'Prefill Batchless CausalLM: {out_path} saved.')
    
    llama_model.layers[0].self_attn.forward = origin_forward
    # exit(0)



def save_decode_batchless(model, save_path = 'model_files/'):
    llama_model = model.model
    
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    

    def batchless_forward(self, query_states, key_states, value_states, cos, sin, attention_mask, past_key , past_value):
        bsz = 1
        assert query_states.shape[0] == bsz
        q_len = int(query_states.shape[-2])
        assert q_len == 1
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        assert past_key is not None
        key_states = torch.cat([past_key, key_states], dim=2) 
        value_states = torch.cat([past_value, value_states], dim=2) 
            
        past_key, past_value = key_states, value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # print(attn_weights.shape, value_states.shape)
        assert attention_mask is not None  # no matter the length, we just slice it
        # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        causal_mask = attention_mask
        # print(causal_mask, attention_mask.shape)
        attn_weights = attn_weights + causal_mask
            
            

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)
        return attn_output, past_key, past_value

    origin_forward = llama_model.layers[0].self_attn.forward
    llama_model.layers[0].self_attn.forward = types.MethodType(batchless_forward, llama_model.layers[0].self_attn)
    
    bsz = 1
    q_len = 1
    past = 5
    q_len_with_past = q_len + past
    past_key_values_length = 0
    hidden_size = llama_model.embed_tokens.weight.shape[-1]
    device = llama_model.device
    query_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    key_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    value_states = torch.zeros((bsz, q_len, hidden_size), dtype=torch.float16, device=device)
    past_key = torch.zeros((1,32, past, hidden_size//32), dtype=torch.float16, device=device)
    past_value = torch.zeros((1,32, past, hidden_size//32), dtype=torch.float16, device=device)
    cos = torch.zeros((bsz, q_len, hidden_size//32), dtype=torch.float16, device=device)
    sin = torch.zeros((bsz, q_len, hidden_size//32), dtype=torch.float16, device=device)
    attention_mask = torch.zeros((1,1, q_len, q_len_with_past), dtype=torch.float16, device=device)

    dynamic_axes={'past_key': {2: 'past'}, 'past_value': {2: 'past'},
                   'attention_mask': {3: 'q_len_with_past'}}
    dynamic_axes.update({'past_key_output':{2: 'q_len_with_past'}, 'past_value_output':{2: 'q_len_with_past'}})
    out_path = os.path.join(save_path,"./batchless_decode.onnx")

    torch.onnx.export(llama_model.layers[0].self_attn,
                    args=(query_states,key_states, value_states, cos, sin, attention_mask, past_key, past_value),
                    f=out_path,
                    opset_version=17,
                    input_names=['query_states','key_states', 'value_states','cos','sin', 'attention_mask',"past_key","past_value"],
                    output_names=['attn_output', 'past_key_output', 'past_value_output'] ,
                    dynamic_axes=dynamic_axes)
    print(f'Decoding Batchless CausalLM: {out_path} saved.')
    
    llama_model.layers[0].self_attn.forward = origin_forward
    
def inference(model, inputs, max_tokens):
    
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
        )

        # 解码生成的文本
        print(outputs[0].shape)
        return outputs[0]

    print(f"{generated_text}")

    
def save_embed_tokens(model, save_dir = 'model_files/'):
    save_name = os.path.join(save_dir, 'embed_tokens.pt')
    tensor = model.model.embed_tokens.weight.requires_grad_(False).data.cpu()
    torch.save(tensor, save_name)
    print(f"Embedding tokens saved to {save_name}")

def main(model: str, output_dir: str = 'model_files/', export: str = None, num_layers: int = 32, test: bool = False, input = "San Francisco is a", device = None, max_tokens = 7):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, attn_implementation='eager', torch_dtype="float16")
    model.eval()

    # Modify the model if num_layers is specified.
    model.model.layers = model.model.layers[:num_layers]

    if not device:
        device = model.model.device
    elif device != "cpu":
        model.to(device)
        device = model.model.device
    # Export the model based on the specified export type
    if export == 'batchful':
        save_batchful(model, num_layers, output_dir, return_last_seq_len=True)
    elif export == 'prefill_batchless':
        save_prefill_batchless(model, output_dir)
    elif export == 'decode_batchless':
        save_decode_batchless(model, output_dir)
    elif export == 'embed_tokens':
        save_embed_tokens(model, output_dir)
    elif export == "tokenizer":
        tokenizer.save_pretrained(output_dir)
    elif test:
        text = input# "San Francisco is a"
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        output = inference(model, inputs, max_tokens)
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"{generated_text}") 
    else:
        print("Invalid export type or test flag not set. Please specify --export or --test.")

if __name__ == '__main__':
    import fire
    fire.Fire(main)