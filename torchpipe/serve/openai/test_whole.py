import torch 
# sequence_length = 5
# target_length = sequence_length
# def generate_mask(sequence_length, target_length):
#     min_dtype = torch.finfo(torch.float16).min
#     dtype = torch.float16   
#     device = 'cuda:0'
#     cache_position = torch.arange(sequence_length, device=device)
#     causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
#     if sequence_length != 1:
#         causal_mask = torch.triu(causal_mask, diagonal=1)
#     causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)

#     print(causal_mask)
# generate_mask(sequence_length, target_length)

import torchpipe
model = torchpipe.Pipe({"backend":"SyncTensor[PrefillAttentionMask]"})
input=  {'data': torch.randn(1, 10, 768).half()}
model(input)
print(input['result'])


from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
print(transformers.__version__)

# 加载模型和分词器
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
text = "San Francisco is a"
inputs = tokenizer(text, return_tensors="pt", return_attention_mask=False)

# 只获取 input_ids
input_ids = inputs["input_ids"]
print(input_ids)
print(inputs)