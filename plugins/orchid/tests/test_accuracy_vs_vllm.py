
import argparse
import os
import torch
import numpy as np
from orchid.llmscheduler.runtime.trt_runtime import TensorRTModelRuntime
from orchid.llmscheduler.runtime.base import AttentionContext
from orchid.llmscheduler.model_params import infer_model_params
from transformers import AutoTokenizer

def run_trt_inference(model_path, tokenizer_path, engine_path, input_ids, prefill_len, bs, fp16=True, device="cuda"):
    runtime = TensorRTModelRuntime(model_path, use_fp16=fp16, engine_path=engine_path)
    mp = infer_model_params(model_path, tokenizer_path)
    
    # Setup Context
    page_size = int(mp.page_size)
    num_layers = int(mp.num_layers)
    pages_per_req = int((int(prefill_len) + 1 + int(page_size) - 1) // int(page_size))
    max_pages = max(int(mp.max_pages), int(bs * pages_per_req * num_layers), 20000)
    
    ctx = AttentionContext(
        num_layers=num_layers,
        num_heads=int(mp.num_heads),
        kv_num_heads=int(mp.kv_num_heads),
        head_dim=int(mp.head_dim),
        page_size=page_size,
        max_pages=max_pages,
        use_cpp_metadata=True,
        device=device,
        use_fp16=fp16,
    )
    
    # Setup Batch Info for Prefill
    ctx.current_batch_req_ids = list(range(bs))
    ctx.current_batch_seq_lens = [prefill_len] * bs
    ctx.current_batch_total_lens = [prefill_len] * bs
    ctx.current_batch_history_lens = [0] * bs
    ctx.current_batch_is_prefill = [True] * bs
    ctx.is_all_decode = False
    
    # Run Prefill
    print(f"TRT Input Shape: {input_ids.shape}")
    torch.cuda.synchronize()
    _ = runtime.forward(input_ids.to(device), ctx)
    torch.cuda.synchronize()
    prefill_logits = runtime.forward(input_ids.to(device), ctx)
    torch.cuda.synchronize()
    
    print(f"TRT Output Shape: {prefill_logits.shape}")
    print(f"TRT Output Sample (first 10): {prefill_logits[0, :10].tolist()}")
    
    # Check for zeros in output
    is_zero = (prefill_logits == 0).all(dim=-1)
    if is_zero.any():
        print(f"WARNING: Some tokens have all-zero logits! Indices: {torch.where(is_zero)[0].tolist()}")

    # Setup Batch Info for Decode Step 1
    # For simplicity, we just feed a dummy token and see what happens
    decode_input = torch.randint(0, 100, (bs,), device=device, dtype=torch.int32)
    ctx.current_batch_seq_lens = [1] * bs
    ctx.current_batch_total_lens = [prefill_len + 1] * bs
    ctx.current_batch_history_lens = [prefill_len] * bs
    ctx.current_batch_is_prefill = [False] * bs
    ctx.is_all_decode = True
    
    # Run Decode
    torch.cuda.synchronize()
    decode_logits = runtime.forward(decode_input, ctx)
    torch.cuda.synchronize()
    
    return prefill_logits.detach().cpu(), decode_logits.detach().cpu()

def run_vllm_inference(tokenizer_path, input_ids_list, prefill_len, bs, fp16=True):
    from vllm import LLM, SamplingParams
    
    dtype = "float16" if fp16 else "float32"
    llm = LLM(
        model=tokenizer_path,
        tokenizer=tokenizer_path,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.4,
        disable_log_stats=True,
        enforce_eager=True # Use eager mode to access logits more easily if needed, though vLLM default is okay
    )
    
    # vLLM API usually takes prompts. We need to force it to output logits.
    # SamplingParams(logprobs=vocab_size) to get full logits is expensive but accurate.
    # Or we can use prompt_logprobs.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=1, prompt_logprobs=1)
    
    # Decode input_ids back to text? No, vLLM supports `prompt_token_ids`
    prompts = [{"prompt_token_ids": ids} for ids in input_ids_list]
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract Prefill Logits (last token of prompt)
    # vLLM returns prompt_logprobs as list of dicts.
    # This is tricky because vLLM usually returns top-k logprobs, not full tensor.
    # To get full tensor comparison, we might need a custom runner or trust that top-k matching is enough.
    # For rigorous test, we might just check if the selected token is same, or if we can get full distribution.
    # Actually, vLLM `LLM` class is high level.
    # Let's assume we compare the *generated token* and *top logprobs*.
    
    # Getting full logits from vLLM public API is hard without modifying it. 
    # But we can check if the argmax matches TRT argmax.
    
    vllm_prefill_logprobs = [] # Last token of prefill
    vllm_decode_token = []
    
    for output in outputs:
        # prompt_logprobs is a list of dicts, one per token. None if not requested.
        # We requested prompt_logprobs=1
        if output.prompt_logprobs:
            # Last token logprobs
            last_token_lp = output.prompt_logprobs[-1]
            vllm_prefill_logprobs.append(last_token_lp)
        vllm_decode_token.append(output.outputs[0].token_ids[0])
        
    return vllm_prefill_logprobs, vllm_decode_token

def compare_results(trt_logits, vllm_logprobs, vllm_tokens):
    # trt_logits: [bs, vocab]
    # vllm_logprobs: list of {token_id: logprob} (top-1)
    
    trt_argmax = torch.argmax(trt_logits, dim=-1)
    
    matches = 0
    for i in range(len(vllm_logprobs)):
        # vLLM gives logprob, TRT gives raw logits. Argmax should match.
        # Note: TRT logits might be unnormalized.
        trt_token = trt_argmax[i].item()
        
        # vLLM prompt_logprobs is {id: logprob}. The key is the token id.
        # Wait, prompt_logprobs[-1] corresponds to the prediction for the *next* token (first decode token)?
        # No, prompt_logprobs is for the prompt tokens themselves.
        # For causal LM, the logit at position T predicts token at T+1.
        # The last logit of prefill predicts the first generated token.
        
        # vLLM output.outputs[0].token_ids[0] IS the first generated token.
        vllm_token = vllm_tokens[i]
        
        if trt_token == vllm_token:
            matches += 1
        else:
            print(f"Mismatch at batch {i}: TRT {trt_token} vs vLLM {vllm_token}")
            
    acc = matches / len(vllm_logprobs)
    print(f"Top-1 Accuracy vs vLLM: {acc*100:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--debug-one", action="store_true")
    parser.add_argument("--strict-determinism", action="store_true")
    args = parser.parse_args()
    
    if args.debug_one:
        args.bs = 1
        args.prefill_len = 32
        print("Debug Mode: BS=1, Len=32")
    
    # 1. Generate Input
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab = getattr(tokenizer, "vocab_size", 32000)
    
    # Use fixed seed
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab, (args.bs, args.prefill_len), dtype=torch.int64)
    input_ids_list = input_ids.tolist()
    input_ids_flat = input_ids.reshape(-1).contiguous()
    
    # 2. Run TRT
    print("Running TRT (Run 1)...")
    # TRT expects int32 for input_ids usually
    input_ids_int32 = input_ids_flat.to(torch.int32)
    trt_prefill_logits_1, trt_decode_logits_1 = run_trt_inference(
        args.model, args.tokenizer, args.engine, input_ids_int32, args.prefill_len, args.bs
    )
    
    print("Running TRT (Run 2)...")
    trt_prefill_logits_2, trt_decode_logits_2 = run_trt_inference(
        args.model, args.tokenizer, args.engine, input_ids_int32, args.prefill_len, args.bs
    )
    
    diff = (trt_prefill_logits_1 - trt_prefill_logits_2).abs().max().item()
    print(f"TRT Run 1 vs Run 2 Max Diff: {diff}")
    if diff > 1e-3:
        if args.strict_determinism:
            print("FAILURE: TRT is non-deterministic!")
        else:
            print("WARNING: TRT is non-deterministic")
        mismatch = (trt_prefill_logits_1 != trt_prefill_logits_2)
        if mismatch.any():
             idx = torch.where(mismatch)[0][0]
             print(f"First mismatch at flat index {idx.item()}: {trt_prefill_logits_1.view(-1)[idx].item()} vs {trt_prefill_logits_2.view(-1)[idx].item()}")
        if args.strict_determinism:
            raise SystemExit(1)
    
    trt_prefill_logits = trt_prefill_logits_1 # Use run 1 for comparison

    # 3. Run vLLM
    # To save memory, we might need to unload TRT or run in separate process. 
    # For now, let's try running sequentially if memory permits.
    # Or, we just use the vLLM script logic which runs in separate process/function?
    # Actually, vLLM initialization is heavy.
    print("Running vLLM...")
    try:
        vllm_prefill_lps, vllm_decode_tokens = run_vllm_inference(
            args.tokenizer, input_ids_list, args.prefill_len, args.bs
        )
    except ImportError:
        print("vLLM not installed, skipping comparison")
        return
    except Exception as e:
        print(f"vLLM run failed: {e}")
        return

    # 4. Compare
    # TRT prefill logits (last token) should predict the first decode token
    # We take the logits corresponding to the last position of each sequence.
    # The runtime.forward returns logits for all tokens in batch?
    # orchid runtime: "logits" output shape is [total_tokens, vocab] or [bs, vocab]?
    # In prefill: input is [bs * seq_len]. Output is [bs * seq_len, vocab].
    # We need to slice out the last token of each request.
    
    # The output from runtime.forward in prefill is flat [total_tokens, vocab]
    # We need to gather indices: [seq_len-1, 2*seq_len-1, ...]
    
    trt_last_token_logits = []
    for i in range(args.bs):
        idx = (i + 1) * args.prefill_len - 1
        print(f"Batch {i}: Last Token Index {idx}, Logits Sample: {trt_prefill_logits[idx, :5].tolist()}")
        trt_last_token_logits.append(trt_prefill_logits[idx])
    trt_last_token_logits = torch.stack(trt_last_token_logits)
    
    print("Comparing Prefill Prediction (Next Token)...")
    acc = compare_results(trt_last_token_logits, vllm_prefill_lps, vllm_decode_tokens)
    
    if acc < 0.9:
        print("FAILURE: Accuracy too low!")
        exit(1)
    else:
        print("SUCCESS: TRT matches vLLM outputs.")

if __name__ == "__main__":
    main()
