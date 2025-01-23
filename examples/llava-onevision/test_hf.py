import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoTokenizer
import os

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
# model_id = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
question = "What are these?"
# question = "How many cats?"

stop_token_ids=0
import os

INFER_BACKEND = "tp" # hf tp vllm
INFER_BACKEND = os.getenv("INFER_BACKEND", INFER_BACKEND)

img_name = "000000039769.jpg"
if not os.path.exists(img_name):
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
else:
    raw_image = Image.open(img_name)
    


def get_prompt(question:str, modality: str):
    if modality == "video":
        prompt = f"<|im_start|>user <video>\n{question}<|im_end|> \
        <|im_start|>assistant\n"

    elif modality == "image":
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
    return prompt
    
if INFER_BACKEND == "hf":
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, 
        attn_implementation = "eager",
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)
    # model.language_model.model._attn_implementation = "eager"

    processor = AutoProcessor.from_pretrained(model_id)

    # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(prompt, type(prompt))

    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    if True:
        # print(inputs["input_ids"].shape, processor)
        # attention_mask = torch.ones(input_ids.shape,dtype=torch.long,device=device)
        # inputs['pad_token_id'] = 0
        print(inputs.keys())
        print(inputs['input_ids'].shape, inputs['attention_mask'].shape)


    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(output[0][:2], " || ", output[0].shape)
    print(processor.decode(output[0][:], skip_special_tokens=True))


if INFER_BACKEND == "vllm":
    
    
    from vllm import LLM, SamplingParams
    prompt = get_prompt(question, "image")
    print(prompt)
    llm = LLM(model=model_id, max_model_len=8192)
    sampling_params = SamplingParams(temperature=0.0,
                                        max_tokens=200,
                                        stop_token_ids=None)
    inputs_one=[]
    inputs_one.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": raw_image
                    },
                })

    outputs = llm.generate(inputs_one, sampling_params=sampling_params)

    print("vllm: \n")
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    
if INFER_BACKEND == "tp":
    import torchpipe
    import torch, os, glob

    TRT_VERSION = '10.7.0.23' # 10.2.0.19

    plugin=torchpipe.utils.cpp_extension.load(name="plugin_ov", sources=glob.glob("cpp/*.cpp")+ glob.glob("cpp/*.cc"),
                                    extra_include_paths=[f'/workspace/TensorRT-{TRT_VERSION}/include/'],
                                    extra_ldflags=[f'-L/workspace/TensorRT-{TRT_VERSION}/targets/x86_64-linux-gnu/lib/','-lnvinfer_plugin','-lnvinfer','-lipipe','-Wl,--no-as-needed', '-lcuda'],
                                    verbose=True,
                                    rebuild_if_exist = True,
                                    is_python_module=False)
    
    
    
    
    prompt = get_prompt(question, "image")
    print(prompt)
    tok = AutoTokenizer.from_pretrained("model_files")
    
    from transformers import AutoImageProcessor
    img_processor = AutoImageProcessor.from_pretrained("model_files")
    # import pdb; pdb.set_trace()
    img = img_processor.preprocess(raw_image)
    img['image_sizes'], len(img['pixel_values']), img['pixel_values'][0].shape
    
    encoded_prompt = tok.encode(prompt)
    
    print(tok, tok.added_tokens_decoder)
    print(tok.eos_token, tok.added_tokens_encoder)
    image_placeholder = tok.added_tokens_encoder["<image>"]
    eos = tok.added_tokens_encoder[tok.eos_token]
    print(image_placeholder, eos, encoded_prompt)
    
    from PackImageFeatures import PackImageFeatures
    torchpipe.register_backend(PackImageFeatures, "PackImageFeatures")
    # from SamplingParams import CSamplingParams
    import RegisterPyBackends
    # torchpipe.register_backend(RegisterPyBackends, "RegisterPyBackends")

    model = torchpipe.pipe("config/model.toml")

    # <|im_end|> {'<|endoftext|>': 151643, '<|im_start|>': 151644, '<|im_end|>': 151645, '<image>': 151646, '<video>': 151647}
    
    inputs_embeds = torch.zeros((2941, 896), dtype=torch.float16).to("cuda")
    index_select = torch.zeros((2,), dtype=torch.long).to(inputs_embeds.device)
    index_select[-1] = 2940
    index_select[-2] = 2939
    # input = {'data' : [inputs_embeds, index_select], 'node_name': 'entry'}
    input = {'request_id':"0", 'data' : torch.tensor(encoded_prompt),"pixel_values":torch.from_numpy(img['pixel_values'][0]), 'node_name': 'entry'}
    # 
    sampling_params = torchpipe._C.TypedDict({"max_tokens": 21, "max_seq_len":4096, "stop_token_ids":[151645]})
    
    input.update({'img_h':img['image_sizes'][0][0],'img_w':img['image_sizes'][0][1], "sampling_params":sampling_params})
    # import pdb; pdb.set_trace()
    
    
    model(input)
    
    # import pdb; pdb.set_trace()
    
    if len(input['result']) == 2:
        
        print(input['result'][0])
        
        print(input['result'][1])
        first_re = input['result'][0]
        print(type(first_re))
        print( (first_re.shape))
        print(first_re[4,0:4])
        # import pdb; pdb.set_trace()
        # print(input['embed_tokens'].shape)
        # torch.load('a.pt').squeeze(0)[3,0], input['result'][3,0]
        print(torch.allclose(torch.load('a.pt').squeeze(0), first_re))
        # a=torch.allclose(torch.load('a.pt').squeeze(0)[0:3], first_re[0:3])
        # b=torch.allclose(torch.load('a.pt').squeeze(0)[-9:], first_re[-9:])

    
    storage = torchpipe.ThreadSafeKVStorage.getInstance(torchpipe._C.SCHEDULER)
    result_queue = storage[('iteration', 'queue')]
    a = result_queue.as_queue()
    
    generated_tokens = []
    while not a.empty():
        data = a.WaitForPop(50)
        if data is None:
            continue
        generated_tokens.append(data["generated_token"])

    re = tok.decode(generated_tokens)
    print(re)
    # import pdb; pdb.set_trace()