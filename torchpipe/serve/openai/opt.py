import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
print(transformers.__version__)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m",attn_implementation='eager')
model.eval()

# 将模型转换为 fp16
model.half()
print(model)

# 将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备输入文本
text = "San Francisco is a"
inputs = tokenizer(text, return_tensors="pt").to(device)
print(model)

 

# 使用 fp16 进行推理
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=7,
        do_sample=False
    )

# 解码生成的文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"{generated_text}")