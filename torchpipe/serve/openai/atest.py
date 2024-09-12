import tiktoken
import timeit
# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.decode(string))
#     print(encoding.decode(string))
#     return num_tokens

# # print(num_tokens_from_string("Hello world, let's test tiktoken.", "cl100k_base"))
# print(num_tokens_from_string([12,12,24,443], "cl100k_base"))
# # print(tiktoken.get_encoding('openai_public'))


from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


model_id = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(type(tokenizer)) #tokenizer
# tokenizer.use_default_system_prompt = True
assert isinstance(tokenizer, PreTrainedTokenizerFast)
def chat_with_llama(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    def encode_prompt():
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

    execution_time = timeit.timeit(encode_prompt, number=100)
    
    # 计算平均每次执行的时间（毫秒）
    avg_time_ms = (execution_time / 100) * 1000

    print(f"平均每次执行时间: {avg_time_ms:.2f}毫秒")
    print((input_ids.shape))
    
    return input_ids
    # input_ids = input_ids.to('cuda')
    # output = model.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2)
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    # return response

# while True:
prompt = 'hello how are u today'*100
response = chat_with_llama(prompt)
# print("Llama:", response)