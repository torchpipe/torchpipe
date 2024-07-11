<!--a10 ~/MiniCPM-V/VILA  docker exec -it debug_mm  bash /workspace/vila/vila-->
<!--27  ~/workspace/mm/VILA/VILA docker exec -it debug_mm  bash-->
<!-- https://g.hz.netease.com/zhangshiyang/vila.git  -->

## visual encoder



In the definetion of class `LlavaMetaForCausalLM` from `./llava/model/llava_arch.py`, put the following code before `image_features = self.encode_images(images).to(self.device)`

```bash
import os
EXPORT_VISUAL_ENCODER = os.environ.get("EXPORT_VISUAL_ENCODER", '0') == "1"
if EXPORT_VISUAL_ENCODER:
    del self.llm.model
    import sys;sys.path.insert(0, './deploy/')
    from export_vila15 import export_vila_visual_encoder
    export_vila_visual_encoder(self, images)
    exit(0)

```
export onnx by:
```bash
#  CUDA_VISIBLE_DEVICES=0,1,2  
 EXPORT_VISUAL_ENCODER=1 python3 -W ignore llava/eval/run_vila.py     --model-path Efficient-Large-Model/VILA1.5-3B      --conv-mode vicuna_v1     --query "<image>\n Please describe the traffic condition."      --image-file "demo_images/av.png" 
```



### BatchfulAttention export {BatchfulAttention}

In `llava/eval/run_vila.py `,
put the following code after `tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)
` 

```bash
import os
EXPORT_Batchful_DECODE = os.environ.get("EXPORT_Batchful_DECODE", '0') == "1"
if EXPORT_Batchful_DECODE:
    import sys;sys.path.insert(0, './deploy/')
    from export_batchable_llama import export_decode_batchful
    export_decode_batchful(model.llm)
    exit(0)
```

Then put the defination of `BatchlessAttention` and `BatchfulAttention` from `export_batchable_llama.py` to `modeling_llama.py`( for example, `/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py`). In LlamaDecoderLayer, replace   
`self.self_attn = LlamaAttention(config=config) (or LlamaFlashAttention2)` with 
`self.self_attn = BatchfulAttention(config=config)`.
Then export model by:

```bash
 CUDA_VISIBLE_DEVICES=3 EXPORT_Batchful_DECODE=1 python3 -W ignore llava/eval/run_vila.py     --model-path Efficient-Large-Model/VILA1.5-3B      --conv-mode vicuna_v1     --query "<image>\n Please describe the traffic condition."      --image-file "demo_images/av.png" 

```

(optinal) test with tensorrt
```bash
 export LD_LIBRARY_PATH=/workspace/TensorRT-10.2.0.19/lib:$LD_LIBRARY_PATH
export PATH=/workspace/TensorRT-10.2.0.19/bin/:$PATH

 CUDA_VISIBLE_DEVICES=3 trtexec --onnx=onnx/decode_batchful.onnx --fp16  \
         --saveEngine=onnx/decode_batchful_2047.trt \
         --shapes=inputs_embeds:2047x2560 #130.441 ms

 CUDA_VISIBLE_DEVICES=3 trtexec --onnx=onnx/decode_batchful.onnx --fp16  \
         --saveEngine=onnx/decode_batchful_244.trt \
         --shapes=inputs_embeds:244x2560 #  20.9047 ms

 CUDA_VISIBLE_DEVICES=3 trtexec --onnx=onnx/decode_batchful.onnx --fp16  \
         --saveEngine=onnx/decode_batchful_1.trt \
         --shapes=inputs_embeds:1x2560 #  #  9.4762 ms
```
 
## decoding
### decoding-BatchlessAttention export {decoding-BatchlessAttention}
In LlamaDecoderLayer, replace   
`self.self_attn = BatchfulAttention(config=config)` with 
`self.self_attn = LlamaAttention(config=config)`.

In 
` llava/eval/run_vila.py `,
put the following code after `tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)
` 

```bash
import os
EXPORT_BatchlessAttention = os.environ.get("EXPORT_BatchlessAttention", '0') == "1"
if EXPORT_BatchlessAttention:
    import sys;sys.path.insert(0, './deploy/')
    from export_batchable_llama import export_vila_decode_batchless
    export_vila_decode_batchless(model.llm.model.layers[0].self_attn)
    exit(0)
```

```python
#   
 CUDA_VISIBLE_DEVICES=3 EXPORT_BatchlessAttention=1 python3 -W ignore llava/eval/run_vila.py     --model-path Efficient-Large-Model/VILA1.5-3B      --conv-mode vicuna_v1     --query "<image>\n Please describe the traffic condition."      --image-file "demo_images/av.png" 

```

(optinal) test with tensorrt

```bash
 export LD_LIBRARY_PATH=/workspace/TensorRT-10.2.0.19/lib:$LD_LIBRARY_PATH
export PATH=/workspace/TensorRT-10.2.0.19/bin/:$PATH

 
CUDA_VISIBLE_DEVICES=5 trtexec --onnx=onnx/batchless.onnx --fp16  \
         --saveEngine=onnx/batchless_2046.trt \
         --shapes=past_key:1x20x2046x128,past_value:1x20x2046x128 # 0.125513 ms

CUDA_VISIBLE_DEVICES=5 trtexec --onnx=onnx/batchless.onnx --fp16  \
         --saveEngine=onnx/batchless_1.trt \
         --shapes=past_key:1x20x1x128,past_value:1x20x1x128 # 0.0113037 ms
```




## Prefilling

### Prefilling-BatchlessAttention {Prefilling-BatchlessAttention}
In LlamaDecoderLayer, make sure
`self.self_attn = LlamaAttention(config=config)`.

In 
` llava/eval/run_vila.py `,
put the following code after `tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)
` 

```bash
import os
EXPORT_PREFILL_Batchless = os.environ.get("EXPORT_PREFILL_Batchless", '0') == "1"
if EXPORT_PREFILL_Batchless:
    import sys;sys.path.insert(0, './deploy/')
    from export_batchable_llama import export_prefill_batchless
    export_prefill_batchless(model.llm.model.layers[0].self_attn)
    exit(0)
```

```python
#   
 CUDA_VISIBLE_DEVICES=3 EXPORT_PREFILL_Batchless=1 python3 -W ignore llava/eval/run_vila.py     --model-path Efficient-Large-Model/VILA1.5-3B      --conv-mode vicuna_v1     --query "<image>\n Please describe the traffic condition."      --image-file "demo_images/av.png" 

```

(optinal) test with tensorrt

```bash
 export LD_LIBRARY_PATH=/workspace/TensorRT-10.2.0.19/lib:$LD_LIBRARY_PATH
export PATH=/workspace/TensorRT-10.2.0.19/bin/:$PATH

 
CUDA_VISIBLE_DEVICES=3 trtexec --onnx=onnx/batchless_prefill.onnx --fp16  \
         --saveEngine=onnx/batchless_prefill_2047.trt \
         --shapes=query_states:1x2047x2560,key_states:1x2047x2560,value_states:1x2047x2560,position_ids:1x2047 # 0.86543 ms

CUDA_VISIBLE_DEVICES=3 trtexec --onnx=onnx/batchless_prefill.onnx --fp16  \
         --saveEngine=onnx/batchless_prefill_244.trt \
         --shapes=query_states:1x244x2560,key_states:1x244x2560,value_states:1x244x2560,position_ids:1x244 # 0.0585938 ms

CUDA_VISIBLE_DEVICES=3 trtexec --onnx=onnx/batchless_prefill.onnx --fp16  \
         --saveEngine=onnx/batchless_prefill_1.trt \
         --shapes=query_states:1x1x2560,key_states:1x1x2560,value_states:1x1x2560,position_ids:1x1 # 0.0111328 ms
```

<!-- ### (optinal) export the whole model
In ` llava/eval/run_vila.py `,
put the following code after `tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)
` 

```bash
import os
EXPORT_PREFILLING = os.environ.get("EXPORT_PREFILLING", '0') == "1"
if EXPORT_PREFILLING:
    import sys;sys.path.insert(0, './deploy/')
    from export_batchable_llama import export_vila_prefilling
    export_vila_prefilling(model.llm, out_dir = 'onnx/prefilling/')
    exit(0)
```
Then export model by:
```bash
CUDA_VISIBLE_DEVICES=3 EXPORT_PREFILLING=1 python3 -W ignore llava/eval/run_vila.py     --model-path Efficient-Large-Model/VILA1.5-3B      --conv-mode vicuna_v1     --query "<image>\n Please describe the traffic condition."      --image-file "demo_images/av.png" 

```
(optinal) check with tensorrt
```bash
# if TensorRT is installed in /workspace/TensorRT-10.2.0.19
export LD_LIBRARY_PATH=/workspace/TensorRT-10.2.0.19:$LD_LIBRARY_PATH
export PATH=/workspace/TensorRT-10.2.0.19/bin/:$PATH

CUDA_VISIBLE_DEVICES=3 trtexec --onnx=onnx/prefilling/prefilling.onnx  --fp16  \
         --saveEngine=onnx/prefilling/prefilling.trt \
         --shapes=inputs_embeds:1x244x2560,position_ids:1x244

CUDA_VISIBLE_DEVICES=3 trtexec --onnx=onnx/prefilling/prefilling.onnx  --fp16  \
         --saveEngine=onnx/prefilling/prefilling_2047.trt \
         --shapes=inputs_embeds:1x2047x2560,position_ids:1x2047
``` -->
