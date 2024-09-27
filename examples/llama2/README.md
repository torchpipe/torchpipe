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
    python run_llama2.py --model model_files/ --input "San Francisco is a" 
    #NUM_LAYER = 2:  San Francisco is a totalitéaletoreignersbyMSран
    #NUM_LAYER = 32:  San Francisco is a city in Northern California that is known
```



# Streaming Inference with Llama2
WIP