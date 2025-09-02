### paramater export 
```bash
# python models/hf_helper.py 

# export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements.txt 
python models/export_onnx_v2.py --num_layers=32 # --model_id=./Llama-2-7b-chat-hf/

```

## run
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install flashinfer-python==0.2.3 # -i https://flashinfer.ai/whl/cu129/torch2.8
# pip install flashinfer-python==0.2.3 --extra-index-url https://flashinfer.ai/whl 
 

rm -rf /opt/hpcx/ncclnet_plugin && ldconfig

export CUDA_VISIBLE_DEVICES=1

python plain_llama2.py --num_layers=32
# num_layers = 2:
# San Francisco is a totalitéaletoreignersbyMSран 
```

## streaming
```bash
python streaming_llama2.py --num_layers=32 --max_num_page=1024

python chat_client.py --prompt="San Francisco is a" --max_tokens 7

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 words" --max_tokens 2048 

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman?" --max_tokens 132  

```
## Benchmark
```bash
 pip install datasets vllm==0.8.4
git clone -b v0.8.4 https://github.com/vllm-project/vllm.git

 pip install datasets vllm==0.8.4

python3 chat_client.py --prompt="Do you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman?" --max_tokens 65 


 img_name=nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3 # triton 2.58.0
docker run --name=exp_vllmclient --runtime=nvidia -e LC_ALL=C -e LANG=C  --ipc=host --cpus=8 --network=host -v `pwd`:/workspace  --shm-size 1G  --ulimit memlock=-1 --ulimit stack=67108864  --privileged=true  -w/workspace -it $img_name /bin/bash

 cd plugins/torchpipe/examples/llama2/
```

```bash
### optional: pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
 pip install pandas datasets
```

```
  python3 vllm/benchmarks/benchmark_serving.py         --backend vllm         --model ./Llama-2-7b-chat-hf/         --dataset-name sharegpt         --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500         --port 8000         --save-result         --result-dir results/         --result-filename vllm_llama7B_tp1_qps_2.json         --request-rate 2
  ```

test on ShareGPT dataset, qps=2, requests=500, llama2-7b, A10-24G

| Option | Median TTFT(ms) | Median TPOT | Input throughput (tok/s) | Output | Generated tokens |
| ------ | --------------- | ----------- | ---------- | ------ | ----- |
| vLLM   | 68              | 51          | 421 | 380    | 105794           |
| our    | 68              | 43          | 423  | 383    | 106175           |
| our/trt-attn  | 138690 | 248          | 160 | 383    | 105702           |

trt-attn:
============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  733.06    
Total input tokens:                      117316    
Total generated tokens:                  105702    
Request throughput (req/s):              0.68      
Input token throughput (tok/s):          160.04    
Output token throughput (tok/s):         144.19    
---------------Time to First Token----------------
Mean TTFT (ms):                          138690.59 
Median TTFT (ms):                        141445.85 
P99 TTFT (ms):                           349376.61 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          596.40    
Median TPOT (ms):                        248.71    
P99 TPOT (ms):                           6425.63   
---------------Inter-token Latency----------------
Mean ITL (ms):                           1009.75   
Median ITL (ms):                         218.95    
P99 ITL (ms):                            978.54    
==================================================

- +Complete page management and batching strategy; +Recomputation when memory is insufficient; +Incremental decoding, +Optimized strategy for memory shortage; +High concurrency result verification

- Feasible solution: Automatically handle linear layers with AI compiler, temporarily delegate attention layer to third-party kernel library (flashinfer); utilize general serving framework for target-agnostic scheduling at upper level



<!-- https://github.com/dreaming-panda/MagicEnc/blob/0d05cec01cdff53d51daa7402fa267595e3bc12b/llama.py#L66 -->


## todo

https://github.com/NVIDIA/TensorRT-LLM/issues/967


> Warning: This is a research project and may not be a production-ready solution. Use at your own risk.


- MagicPiG
- FastDecode
- PipeThreader: Software-Defined Pipelining for Efficient DNN Execution (osdi25)
- NeuStream: Bridging Deep Learning Serving and Stream Processing
https://dl.acm.org/doi/pdf/10.1145/3689031.3717489
- https://chengyupku.github.io/publications/
https://yangzhihome.github.io/
- A SYSTEM FOR MICROSERVING OF LLMS
- https://www.themoonlight.io/en/review/a-system-for-microserving-of-llms



todo:

修复时间戳，将recomputation加在最前面