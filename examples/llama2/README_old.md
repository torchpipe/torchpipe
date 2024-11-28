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
<!-- 
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
``` -->



# Streaming Inference with Llama2
WIP


```

 

python run_llama2_streaming.py 

python chat_client.py --prompt="Do you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 wordsDo you know the book Traction by Gino Wickman? generate anwser >= 2000 words" --max_tokens 2048 


python chat_client.py --prompt="San Francisco is a" --max_tokens 132
python chat_client.py --prompt="Do you know the book Traction by Gino Wickman?" --max_tokens 132  


 python3 benchmark_serving.py --backend vllm  --model ../Llama-2-7b-chat-hf/         --dataset-name sharegpt --dataset-path ../ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500 --port 8080 --save-result --result-dir results/ --result-filename vllm_llama7B_tp1_qps_2.json --request-rate 2   
```

```
============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  52.67     
Total input tokens:                      14056     
Total generated tokens:                  10508     
Request throughput (req/s):              0.95      
Input token throughput (tok/s):          266.86    
Output token throughput (tok/s):         199.50    
---------------Time to First Token----------------
Mean TTFT (ms):                          121.19    
Median TTFT (ms):                        94.22     
P99 TTFT (ms):                           289.86    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          48.93     
Median TPOT (ms):                        48.89     
P99 TPOT (ms):                           84.22     
---------------Inter-token Latency----------------
Mean ITL (ms):                           54.84     
Median ITL (ms):                         52.61     
P99 ITL (ms):                            206.89    
==================================================
```



# Benchmark



### vllm results(qps=2, A10)

```bash
python3 -m vllm.entrypoints.openai.api_server -tp 1 -pp 1 --gpu-memory-utilization 0.95         --model ../Llama-2-7b-chat-hf/ --port 8000 --disable-log-stats --disable-log-requests 

 python3 benchmark_serving.py --backend vllm  --model ../Llama-2-7b-chat-hf/         --dataset-name sharegpt --dataset-path ../ShareGPT_V3_unfiltered_cleaned_split.json         --num-prompts 500 --port 8000 --save-result --result-dir results/ --result-filename vllm_llama7B_tp1_qps_2.json --request-rate 2   
```


```bash
Traffic request rate: 2.0
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [04:38<00:00,  1.80it/s]
============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  278.41    
Total input tokens:                      117316    
Total generated tokens:                  105926    
Request throughput (req/s):              1.80      
Input token throughput (tok/s):          421.38    
Output token throughput (tok/s):         380.47    
---------------Time to First Token----------------
Mean TTFT (ms):                          89.52     
Median TTFT (ms):                        68.46     
P99 TTFT (ms):                           369.73    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          52.80     
Median TPOT (ms):                        52.49     
P99 TPOT (ms):                           74.19     
---------------Inter-token Latency----------------
Mean ITL (ms):                           53.34     
Median ITL (ms):                         44.69     
P99 ITL (ms):                            246.41    
==================================================



qps=2============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  47.79     
Total input tokens:                      14056     
Total generated tokens:                  9480      
Request throughput (req/s):              1.05      
Input token throughput (tok/s):          294.13    
Output token throughput (tok/s):         198.37    
---------------Time to First Token----------------
Mean TTFT (ms):                          107.79    
Median TTFT (ms):                        68.15     
P99 TTFT (ms):                           385.09    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.80     
Median TPOT (ms):                        43.83     
P99 TPOT (ms):                           74.82     
---------------Inter-token Latency----------------
Mean ITL (ms):                           44.27     
Median ITL (ms):                         40.24     
P99 ITL (ms):                            206.77    
==================================================


qps=4============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  38.80     
Total input tokens:                      14056     
Total generated tokens:                  9480      
Request throughput (req/s):              1.29      
Input token throughput (tok/s):          362.30    
Output token throughput (tok/s):         244.35    
---------------Time to First Token----------------
Mean TTFT (ms):                          105.80    
Median TTFT (ms):                        67.86     
P99 TTFT (ms):                           309.15    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          51.83     
Median TPOT (ms):                        49.12     
P99 TPOT (ms):                           100.84    
---------------Inter-token Latency----------------
Mean ITL (ms):                           48.01     
Median ITL (ms):                         42.77     
P99 ITL (ms):                            260.83    
==================================================
```



