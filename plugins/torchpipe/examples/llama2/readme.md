### paramater export 
```bash
# python models/hf_helper.py 

# export HF_ENDPOINT=https://hf-mirror.com
python models/export_onnx_v2.py --num_layers=2 # --model_id=/benchmark/Llama-2-7b-chat-hf/
```

## run
```
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install flashinfer-python -i https://flashinfer.ai/whl/cu128/torch2.7


python plain_llama2.py
# num_layers = 2:
# San Francisco is a totalitéaletoreignersbyMSран 
```

## streaming
```
python streaming_llama2.py --num_layers=32

python chat_client.py --prompt="San Francisco is a" --max_tokens 7

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 words" --max_tokens 2048 

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman?" --max_tokens 132  

```
## Benchmark

test on ShareGPT dataset, qps=2, requests=500, llama2-7b, A10-24G

| Option | Median TTFT(ms) | Median TPOT | Input throughput (tok/s) | Output | Generated tokens |
| ------ | --------------- | ----------- | ------------------------ | ------ | ---------------- |
| vLLM   | 68              | 51          | 421                      | 380    | 105794           |
| our    | 68              | 43          | 423                      | 383    | 106175           |

- +Complete page management and batching strategy; +Recomputation when memory is insufficient; +Incremental decoding, +Optimized strategy for memory shortage; +High concurrency result verification

- Feasible solution: Automatically handle linear layers with AI compiler, temporarily delegate attention layer to third-party kernel library (flashinfer); utilize general serving framework for target-agnostic scheduling at upper level



<!-- https://github.com/dreaming-panda/MagicEnc/blob/0d05cec01cdff53d51daa7402fa267595e3bc12b/llama.py#L66 -->


## todo

https://github.com/NVIDIA/TensorRT-LLM/issues/967