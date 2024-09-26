# Non-Streaming Inference with Llama2
## Prepare llama2 models
> Only test on transformers=4.44.2
    
```bash
    pip install transformers==4.44.2 fire
    

    export NUM_LAYER=32 # set to 2 if using 2-layer model for debug on 12GB-GPU etc.
    # inference
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --test --num_layers $NUM_LAYER
    # San Francisco is a totalitéaletoreignersbyMSран (if number of layers is 2)
```

## Export llama2 models
```bash
# batchful part
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export batchful --num_layers $NUM_LAYER --input "San Francisco is a"

# batchless part
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export prefill_batchless --num_layers $NUM_LAYER2

    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export decode_batchless --num_layers $NUM_LAYER

# export embed_tokens.pt
    python export_llama2.py --model meta-llama/Llama-2-7b-chat-hf --output_dir model_files/ --export embed_tokens
```

## run inference
```bash
# tensorrt engine will be automatically generated and cached. Please make sure there are enough GPU memory, or you can generate the engines multiple times.
    python run_llama2.py --model meta-llama/Llama-2-7b-chat-hf --input "San Francisco is a" 
    #NUM_LAYER = 2:  San Francisco is a totalitéaletoreignersbyMSран
```