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


```bash
qps=2, torchpipe
============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  277.20    
Total input tokens:                      117316    
Total generated tokens:                  119503    
Request throughput (req/s):              1.80      
Input token throughput (tok/s):          423.21    
Output token throughput (tok/s):         431.10    
---------------Time to First Token----------------
Mean TTFT (ms):                          95.50     
Median TTFT (ms):                        69.14     
P99 TTFT (ms):                           623.87    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.77     
Median TPOT (ms):                        39.28     
P99 TPOT (ms):                           57.79     
---------------Inter-token Latency----------------
Mean ITL (ms):                           45.88     
Median ITL (ms):                         41.20     
P99 ITL (ms):                            179.10    
==================================================

```

<!-- https://github.com/dreaming-panda/MagicEnc/blob/0d05cec01cdff53d51daa7402fa267595e3bc12b/llama.py#L66 -->


## todo

https://github.com/NVIDIA/TensorRT-LLM/issues/967