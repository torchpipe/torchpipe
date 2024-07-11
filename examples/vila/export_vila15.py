import torch
from torch import nn
import os

class VisualWrapper(torch.nn.Module):
    def __init__(self, model, trace_function, need_fix_opset =False):
        super().__init__()
        model.eval()
        self.model = model
        self.trace_function = trace_function
        
        # a fix for valid conv2d: https://github.com/pytorch/pytorch/pull/89107
        # if torch >= 2.1 used, no need to fix. But VILA need torch 2.0
        # replace: find / -name symbolic_opset9.py 
        # with: https://raw.githubusercontent.com/shubhambhokare1/pytorch/7c076cae4eee3d7a316231f5a1e261d1626869aa/torch/onnx/symbolic_opset9.py
        
        
        if need_fix_opset and torch.__version__ <= '2.0.1z':
            # get python version and check if python3.10 is used
            import sys
            python_version = sys.version_info
            pv = f'{python_version.major}.{python_version.minor}'
            
            src_url = f"https://raw.githubusercontent.com/shubhambhokare1/pytorch/7c076cae4eee3d7a316231f5a1e261d1626869aa/torch/onnx/symbolic_opset9.py"
            # from 'find / -name symbolic_opset9.py', not all symbolic_opset9.py are the same 
            tar_path = f"/usr/local/lib/python{pv}/dist-packages/torch/onnx/symbolic_opset9.py"
            if os.path.exists(tar_path):
                s = f'rm /usr/local/lib/python{pv}/dist-packages/torch/onnx/__pycache__/symbolic_opset9.cpython-310.pyc'
                os.system(f"wget {src_url} -O {tar_path}")
                os.system(s)
            
            src_url = "https://raw.githubusercontent.com/shubhambhokare1/pytorch/7c076cae4eee3d7a316231f5a1e261d1626869aa/torch/onnx/symbolic_opset13.py"
            # from 'find / -name symbolic_opset13.py', not all symbolic_opset13.py are the same 
            tar_path = f"/usr/local/lib/python{pv}/dist-packages/torch/onnx/symbolic_opset13.py"
            if os.path.exists(tar_path):
                s = f'rm /usr/local/lib/python{pv}/dist-packages/torch/onnx/__pycache__/symbolic_opset13.cpython-310.pyc'
                os.system(f"wget {src_url} -O {tar_path}")
                os.system(s)
        
    def forward(self, input):
        return getattr(self.model,self.trace_function)(input)

def export_vila_visual_encoder(self, images):
    dynamic_axes={'input': {0: 'x'}, 'output': {0: 'x'}}
    model = VisualWrapper(self, 'encode_images', need_fix_opset = True).eval()
    os.makedirs('onnx/', exist_ok=True)
    out_path = "onnx/visual_encoder.onnx"
    torch.onnx.export(model,
                      images,
                      out_path,
                      opset_version=17,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=dynamic_axes)
    print(f'{out_path} saved.')
    

   
def export_vila_prefill(self, model_inputs, logits_processor, logits_warper, input_ids):
    class Wrapper(torch.nn.Module):
        def __init__(self, llavallamamodel):
            super().__init__()
            self.llavallamamodel = llavallamamodel
            # device = self.llavallamamodel.llm.lm_head.weight.data.device
            # dtype = self.llavallamamodel.llm.lm_head.weight.data.dtype
    
            # temperature = generation_kwargs['temperature']
            # self.llavallamamodel.llm.lm_head.weight.data /= temperature
            # self.llavallamamodel.llm.lm_head.weight.data = self.llavallamamodel.llm.lm_head.weight.data.to(dtype)
            
        def forward(self, inputs_embeds, attention_mask, position_ids):
            model_inputs['inputs_embeds'] = inputs_embeds
            model_inputs['attention_mask'] = attention_mask
            model_inputs['position_ids'] = position_ids
            # input_ids = model_inputs['input_ids']
            outputs = self.llavallamamodel(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            past_key_values = outputs['past_key_values']
            
            kv = [item for pair in past_key_values for item in pair]
            kv = torch.cat(kv, dim=1)


            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            return probs, kv
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # return next_tokens
    model = Wrapper(self).eval()
    
    inputs_embeds = model_inputs['inputs_embeds']
    position_ids = model_inputs['position_ids']
    attention_mask = model_inputs['attention_mask']
    
    dynamic_axes={'inputs_embeds': {1: 'seq_len'}, 'attention_mask': {1: 'seq_len'}, 'position_ids': {1: 'seq_len'},'kv': {2: 'seq_len'}}
    
    out_path = "onnx/prefill.onnx"

    torch.onnx.export(model,
                      (inputs_embeds, attention_mask, position_ids),
                      out_path,
                      opset_version=17,
                      input_names=['inputs_embeds', 'attention_mask', 'position_ids'],
                      output_names=['output'] + ['kv'],
                      dynamic_axes=dynamic_axes)
    print(f'{out_path} saved.')


def export_vila_decode(self, model_inputs, logits_processor, logits_warper, total_input_ids):
    class Wrapper(torch.nn.Module):
        def __init__(self, llavallamamodel):
            super().__init__()
            self.llavallamamodel = llavallamamodel
            # device = self.llavallamamodel.llm.lm_head.weight.data.device
            # dtype = self.llavallamamodel.llm.lm_head.weight.data.dtype
    
            # temperature = generation_kwargs['temperature']
            # self.llavallamamodel.llm.lm_head.weight.data /= temperature
            # self.llavallamamodel.llm.lm_head.weight.data = self.llavallamamodel.llm.lm_head.weight.data.to(dtype)
            
        def forward(self, input_ids, attention_mask, position_ids, past_key_values):
            model_inputs['input_ids'] = input_ids
            model_inputs['attention_mask'] = attention_mask
            model_inputs['position_ids'] = position_ids
            
            
            new_kv = []
            for i in range(32):
                k = past_key_values[:,40*i:40*i+20,:,:]
                v = past_key_values[:,40*i+20:40*i+40,:,:]
                print(k.shape,v.shape)
                new_kv.append((k,v))
            model_inputs['past_key_values']  = new_kv
            # input_ids = model_inputs['input_ids']
            outputs = self.llavallamamodel(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            past_key_values_out = outputs['past_key_values']
            
            kv = [item for pair in past_key_values_out for item in pair]
            kv = torch.cat(kv, dim=1)


            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(total_input_ids, next_token_logits)
            next_token_scores = logits_warper(total_input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            return probs, kv
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # return next_tokens
    model = Wrapper(self).eval()
    
    input_ids = model_inputs['input_ids']
    position_ids = model_inputs['position_ids']
    attention_mask = model_inputs['attention_mask']
    assert attention_mask is not None
    print(attention_mask.shape, 'attention_mask')
    past_key_values = model_inputs['past_key_values']
    kv = [item for pair in past_key_values for item in pair]
    past_key_values = torch.cat(kv, dim=1)
    # assert past_key_values is not None
    
    dynamic_axes={'past_key_values': {2: 'seq_len'}, 'attention_mask': {1: 'seq_len_plus_1'}, 'kv': {2: 'seq_len_plus_1'}}
    
    out_path = "onnx/decode.onnx"

    torch.onnx.export(model,
                      (input_ids, attention_mask, position_ids, past_key_values),
                      out_path,
                      opset_version=17,
                      input_names=['input_ids', 'attention_mask', 'position_ids', 'past_key_values'],
                      output_names=['output'] + ['kv'],
                      dynamic_axes=dynamic_axes)
    print(f'{out_path} saved.')
