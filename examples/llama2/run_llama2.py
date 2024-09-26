
import torchpipe
import torch, os, glob



plugin=torchpipe.utils.cpp_extension.load(name="plugin", sources=glob.glob("cpp/*.cpp"),
                                   extra_include_paths=['/workspace/TensorRT-10.2.0.19/include/'],
                                   extra_ldflags=['-L/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/lib/','-lnvinfer_plugin','-lnvinfer','-lipipe','-Wl,--no-as-needed'],
                                   verbose=True,
                                   is_python_module=False)
storage = torchpipe.ThreadSafeKVStorage.getInstance()
  
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(model: str, input: str):
    tp_model = torchpipe.Pipe("config/llama2.toml")

    tokenizer = AutoTokenizer.from_pretrained(model)
    inputs = tokenizer(input, return_tensors="pt")

    print(inputs["input_ids"])

    inputs = {
        'data': inputs["input_ids"][0],
        'request_id': 'r0',
        'node_name': 'input',
        'trt_plugin': 'batchless_prefill'
    }

    tp_model(inputs)
    print(inputs['result'].shape, inputs.keys())
    print(len(inputs['other']))
    for item in inputs['other']:
        print(item.shape)
    print(inputs['input_tokens_result'])

    out = torch.cat(inputs['input_tokens_result'][1:], dim=0)
    result = tokenizer.decode(out, skip_special_tokens=True)
    print(f"{input} {result}")
 
if __name__ == '__main__':
    import fire
    fire.Fire(main)