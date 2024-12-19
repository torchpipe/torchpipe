### (WIP) llava-onevision
> Note: this is a work in progress and experimental.

#### run llama-ov in transformers:
```bash
python test_hf.py

```

### 导出相关参数

```bash
## 导出 vision_tower.onnx
python models/exporting_llava_onevision.py --action=vision_tower 

## 导出 image_newline.pt
python models/exporting_llava_onevision.py --action=image_newline

## 导出 embed_tokens.pt
python models/exporting_llava_onevision.py --action=embed_tokens

## 导出tokenizer:
python models/exporting_llava_onevision.py --action=tokenizer
```

language model part：
```bash
## 导出 batchable.onnx
python models/exporting_llava_onevision.py --action=batchable

python models/exporting_llava_onevision.py --action=batchless_prefill

 

``` 

### Profile
```bash
/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/bin/trtexec  --onnx=model_files/batchable.onnx  --shapes=inputs_embeds:2941x896,logits_to_keep_mask:2941 --fp16

## set min opt max:
/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/bin/trtexec --onnx=model_files/batchable.onnx    --minShapes=inputs_embeds:1x896,index_select:1 \
--optShapes=inputs_embeds:2940x896,index_select:128  --maxShapes=inputs_embeds:2942x896,index_select:256 

export PATH=/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/bin:$PATH

# trt 10.2
trtexec --onnx=model_files/vision_tower.onnx --fp16 --shapes=input:32x3x384x384 # 272.136
trtexec --onnx=model_files/vision_tower.onnx --fp16 --shapes=input:16x3x384x384 # 136.198
trtexec --onnx=model_files/vision_tower.onnx --fp16 --shapes=input:8x3x384x384 # 68.7783
trtexec --onnx=model_files/vision_tower.onnx --fp16 --shapes=input:4x3x384x384 # 37.2378
trtexec --onnx=model_files/vision_tower.onnx --fp16 --shapes=input:2x3x384x384 # 
trtexec --onnx=model_files/vision_tower.onnx --fp16 --shapes=input:1x3x384x384 # 10.4409
# --maxShapes=inputs_embeds:2942x896,logits_to_keep_mask:2942    \
# --shapes=inputs_embeds:2941x896,logits_to_keep_mask:2941
```




### vllm

https://github.com/vllm-project/vllm/issues/4194#issuecomment-2102487467
https://github.com/vllm-project/vllm/pull/5964

See https://github.com/vllm-project/vllm/pull/8486
```bash

```


## TODO
- [] 检查benchmark_serving 对于VLM和onevision的客户端接口
- []  