# Non-Streaming Inference with Llama2

> WARNING
This project is still in the experimental stage. Do not use it in production environments. 

<details>
<summary>Goal</summary>
The final goal is that, we want serve LLM mainly with TensorRT, but with no dependency on TensorRT-LLM or Triton server. We segment layers based on whether they can be batched with respect to the sequence length's dimension. The model is divided into two parts: batchful and batchless. Model parameters are (mainly) located in the batchful part, whereas the batchless part consists of positional encoding and parameter-free self-attention. After masking the batchless part, we perform a complete trace.

Traditional dynamic batching can be applied the batchful part. We isolate the batchless part as a separate custom sub-graph/(function in future) and implement it using a TensorRT plugin. This plugin does nothing but direct the batchless part to a dedicated TorchPipe server. The management and resource(e.g. kvcache) control operate entirely independently of TensorRT.

The computation for the batchless part could be implemented as a standalone CUDA kernel. However, for simplicity, we have chosen to trace and implement it using TensorRT. TensorRT may internally optimize computations by matching flash attention patterns. The verbose information from TensorRT indicates that it has identified and reassigned Myelin backends for Self-Attention nodes (i.e., /MatMul_1, /Softmax, /MatMul).
</details>

## Prepare llama2 models
> Only test on transformers=4.44.2
    
```bash
    pip install transformers==4.44.2 fire
    pip install --upgrade torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
    

    export NUM_LAYER=32 # set to 2 if using 2-layer model for debug on 12GB-GPU.
    # inference
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --input "San Francisco is a" --test --num_layers $NUM_LAYER 
    #NUM_LAYER = 2: San Francisco is a totalitéaletoreignersbyMSран
    #NUM_LAYER = 32:  San Francisco is a city in Northern California that is known
```

## Export llama2 models
> These steps can be done on CPU (by remove `--device cuda`) if you do not have enough GPU memory.

```bash
# batchful part
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export batchful --num_layers $NUM_LAYER --device cuda
    ## you can use cpu for export, but it will be slow.

# batchless part
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export prefill_batchless  

    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export decode_batchless  

# export embed_tokens.pt
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export embed_tokens

# copy tokenizer from cached huggingface model
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export tokenizer

```

## run inference
```bash
# tensorrt engine will be automatically generated and cached. Please make sure there are enough GPU memory, or you can generate the engines multiple times.
    export CUDA_VISIBLE_DEVICES=1

    # tensorrt >= 10.5 needed
    python run_llama2.py --model model_files/ --input "San Francisco is a" 
    #NUM_LAYER = 2:  San Francisco is a totalitéaletoreignersbyMSран
    #NUM_LAYER = 32:  San Francisco is a city in Northern California that is known


    # python run_llama2.py --model model_files/ --input "Do you know the book Traction by Gino Wickman" --max_tokens 132
    
#     Do you know the book Traction by Gino Wickman?

# Traction is a book written by Gino Wickman, a business coach and author, that provides a framework for creating a successful business. The book focuses on the importance of having a clear vision, establishing a strong leadership team, and implementing a set of core values that guide decision-making.

# The book introduces the concept of the "EOS (Entrepreneurial Operating System)," which is a set of tools and processes that help businesses achieve their goals and create a sustainable, successful organization. The EOS framework includes six key components:

# 1. Vision: Develop
```



# Streaming Inference with Llama2
WIP


```

only support partiqular model and prompt for now.
 

python run_llama2_streaming.py 

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 words" --max_tokens 2048 

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman?" --max_tokens 132  


 python3 benchmark_serving.py --backend vllm  --model ../Llama-2-7b-chat-hf/         --dataset-name sharegpt --dataset-path ../ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 50 --port 8080 --save-result --result-dir results/ --result-filename vllm_llama7B_tp1_qps_2.json --request-rate 2   


============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  71.48     
Total input tokens:                      14056     
Total generated tokens:                  10791     
Request throughput (req/s):              0.70      
Input token throughput (tok/s):          196.63    
Output token throughput (tok/s):         150.96    
---------------Time to First Token----------------
Mean TTFT (ms):                          103.45    
Median TTFT (ms):                        71.23     
P99 TTFT (ms):                           290.93    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          75.08     
Median TPOT (ms):                        75.62     
P99 TPOT (ms):                           109.22    
---------------Inter-token Latency----------------
Mean ITL (ms):                           84.10     
Median ITL (ms):                         82.84     
P99 ITL (ms):                            247.92    
==================================================
```



# Benchmark



### vllm results(qps=2, A10)

```bash
python3 -m vllm.entrypoints.openai.api_server -tp 1 -pp 1 --gpu-memory-utilization 0.95         --model ../Llama-2-7b-chat-hf/ --port 8000 --disable-log-stats --disable-log-requests 

 python3 benchmark_serving.py --backend vllm  --model ../Llama-2-7b-chat-hf/         --dataset-name sharegpt --dataset-path ../ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 50 --port 8000 --save-result --result-dir results/ --result-filename vllm_llama7B_tp1_qps_2.json --request-rate 2   
```

```bash
============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  277.76    
Total input tokens:                      117316    
Total generated tokens:                  105842    
Request throughput (req/s):              1.80      
Input token throughput (tok/s):          422.36    
Output token throughput (tok/s):         381.05    
---------------Time to First Token----------------
Mean TTFT (ms):                          98.79     
Median TTFT (ms):                        69.11     
P99 TTFT (ms):                           541.40    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          52.21     
Median TPOT (ms):                        51.64     
P99 TPOT (ms):                           73.41     
---------------Inter-token Latency----------------
Mean ITL (ms):                           52.88     
Median ITL (ms):                         44.33     
P99 ITL (ms):                            239.62    
==================================================

```



