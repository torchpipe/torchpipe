
## VILA1.5-3B example
In this example, we serve VILA1.5-3B(fp16) with torchpipe, with no dependency on TensorRT-LLM or Triton server. We segment layers based on whether they can be batched with respect to the sequence length's dimension. The model is divided into two parts: batchful and batchless. Model parameters are (mainly) located in the batchful part, whereas the batchless part consists of positional encoding and parameter-free self-attention.  After masking the batchless part, we perform a complete trace. 

Traditional dynamic batching can be applied the batchful part. We isolate the batchless part as a separate custom sub-graph/([function](https://github.com/gramalingam/onnx/blob/main/docs/IR.md#functions) in future) and implement it using a TensorRT plugin. This plugin does nothing but  direct the batchless part to a dedicated TorchPipe server. The management and resource(e.g. kvcache) control operate entirely independently of TensorRT.


The computation for the batchless part could be implemented as a standalone CUDA kernel. However, for simplicity, we have chosen to trace and implement it using TensorRT. TensorRT may internally optimize computations by matching [flash attention patterns](https://github.com/NVIDIA/TensorRT/issues/3647#issuecomment-2054441577). The verbose information from TensorRT indicates that it has identified and reassigned Myelin backends for Self-Attention nodes (i.e., /MatMul_1, /Softmax, /MatMul).

### Features:
- [x] A TensorRT and trace based solution with no need for `TensorRT-LLM` and `Triton inference server`.
- [x] flash attention
- [x] contiguous batching && load banlancing
- [x] ~~PagedAttention~~ memory pool(need further thought)

### limitations:
- tokenizer is not included in C++/torchpipe. Only LlamaForCausalLM is finished yet.

### Run && Benchmark
```
export TENSORRT_PATH=/workspace/TensorRT-10.2.0.19
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/workspace/TensorRT-10.2.0.19/targets/x86_64-linux-gnu/lib/:$LD_LIBRARY_PATH

# RUN only LlamaForCausalLM: 
# prepare the model files and input embeddings (seq_lenX2560)
DEBUG=1 python run_vila.py

# RUN visual part (WIP):
```

## Requirements
- Ensure that you have successfully set up the environment for [VILA](https://github.com/NVlabs/VILA). Follow the instructions in their README carefully. Once successed, you can run the following commands to check.

```bash
python3 -W ignore llava/eval/run_vila.py     --model-path Efficient-Large-Model/VILA1.5-3B      --conv-mode vicuna_v1     --query "<image>\n Please describe the traffic condition."      --image-file "demo_images/av.png" 
```

Follow the following steps to export a few model files.

Assuming you are in the root directory of [VILA](https://github.com/NVlabs/VILA). Copy the following files to the `deploy` directory.
```bash
mkdir ./deploy/

cp /path/to/torchpipe/examples/vila/*.py ./deploy/
```

## batchful part

After masking the batchless part, we export an ONNX model that can be batched, with the sequence length dimension serving as the batch dimension. 

see [BatchfulAttention export](model_exported.md#batchfulattention-export).


Model inputs: inputs_embeds

Model outputs: logits




## Batchless part for Prefilling/Context stage
Model inputs: query_states,key_states, value_states,position_ids

see [prefill export](model_exported.md#prefilling-batchlessattention).

## Batchless part for Decoding stage

Model inputs: query_states,key_states, value_states,position_ids,past_key,past_value

see [batchless part for Decoding](model_exported.md#decoding-batchlessattention).


## (WIP)Visual encoder

Get `onnx/visual_encoder.onnx` file by [exporting visual encoder](model_exported.md#visual-encoder). You can also get it from [build_visual_engine.py](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal)

You need TensorRT 10 to run `run_vila.py`

## (WIP) scheduling

## (WIP)optinal:  Independent prefilling stage(decoupled mode of Prefilling and Decoding)

decoupling of Prefilling and Decoding:

 | Decoupling of Prefilling and Decoding       | Description                                                                                                                         |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Advantages   | - During the prefilling stage, the sequence length dimension is inherently batched.  Engineering can be simplified.  Different GPU numbers can be allocated to different stages for better balancing. |
| Disadvantages| - Model parameters cannot be shared between the Prefilling and Decoding stages, leading to increased GPU memory usage. This may be unacceptable in some scenarios. |


 see [decode export](model_exported.md#decoding).
 

## inference
```python
```


## Reference
- TensorRT-LLM
- [TensorRT-Custom-Plugin-Example](https://github.com/leimao/TensorRT-Custom-Plugin-Example)
- [trt or trt-llm](https://github.com/NVIDIA/TensorRT/issues/3647#issuecomment-2054441577)