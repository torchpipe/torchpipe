import tvm_ffi
import torch
import os
import time
from .flashinfer_backend import kv_append, plan_wrappers, run_decode, run_prefill
import flashinfer

if not bool(int(os.environ.get("LLMSCHEDULER_QUIET", "0"))):
    print("Using Python-only implementation of CompositeAttention plugin")

CURRENT_CTX = None

_PROFILE_ON = bool(int(os.environ.get("LLMSCHEDULER_PROFILE", "0")))
_PROFILE_DETAIL = _PROFILE_ON and bool(int(os.environ.get("LLMSCHEDULER_PROFILE_DETAIL", "0")))
_PROFILE_DETAIL_ALL = _PROFILE_DETAIL and bool(int(os.environ.get("LLMSCHEDULER_PROFILE_DETAIL_ALL_LAYERS", "0")))

def set_context(ctx, stream: int | None = None):
    global CURRENT_CTX
    CURRENT_CTX = ctx
    if ctx is not None and stream is not None:
        s = int(stream)
        prev = getattr(ctx, "_trt_cuda_stream_cached", None)
        if prev != s:
            setattr(ctx, "_trt_cuda_stream_cached", s)
            try:
                setattr(ctx, "_trt_external_stream", torch.cuda.ExternalStream(s))
            except Exception:
                setattr(ctx, "_trt_external_stream", None)

def dtype_code_to_typestr(dtype_code: int) -> str:
    if dtype_code == 0: return "<f4"
    if dtype_code == 1: return "<f2"
    return "<f2"

class CudaWrapper:
    __slots__ = ('__cuda_array_interface__',)
    def __init__(self, ptr, shape, dtype):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": dtype,
            "version": 3
        }

def _wrap_cuda_tensor(ctx, ptr: int, shape: tuple[int, ...], typestr: str, device: str):
    d = ctx.device if ctx is not None else device
    return torch.as_tensor(CudaWrapper(ptr, shape, typestr), device=d)

@tvm_ffi.register_global_func("llmscheduler.composite_attention_impl")
def composite_attention_impl(q_ptr, k_ptr, v_ptr, out_ptr, total_tokens, num_heads, head_dim, kv_num_heads, layer_idx, rope_theta, pos_encoding_mode, dtype_code_in, dtype_code_out):
    if bool(int(os.environ.get("LLMSCHEDULER_ATTENTION_BYPASS", "0"))):
        return

    ctx = CURRENT_CTX
    if ctx is None:
        return

    prof = getattr(ctx, "_prof", None) if _PROFILE_ON else None
    prof_detail = False
    if prof is not None:
        prof_detail = _PROFILE_DETAIL and (_PROFILE_DETAIL_ALL or layer_idx == 0)
    
    prof_on = (prof is not None)
    t_attn0 = time.perf_counter() if prof_on else 0.0

    dtype_str_in = dtype_code_to_typestr(dtype_code_in)
    dtype_str_out = dtype_code_to_typestr(dtype_code_out)

    if prof is not None and layer_idx == 0:
        prof["attn_dtype_in"] = float(dtype_code_in)
        prof["attn_dtype_out"] = float(dtype_code_out)

    device = ctx.device
    
    t_wrap0 = time.perf_counter() if prof_detail else 0.0
    
    q = _wrap_cuda_tensor(ctx, q_ptr, (total_tokens, num_heads, head_dim), dtype_str_in, device)
    k = _wrap_cuda_tensor(ctx, k_ptr, (total_tokens, kv_num_heads, head_dim), dtype_str_in, device)
    v = _wrap_cuda_tensor(ctx, v_ptr, (total_tokens, kv_num_heads, head_dim), dtype_str_in, device)
    out = _wrap_cuda_tensor(ctx, out_ptr, (total_tokens, num_heads, head_dim), dtype_str_out, device)

    if prof_detail and prof is not None:
        prof["attn_wrap_ms"] = float(prof.get("attn_wrap_ms", 0.0)) + float((time.perf_counter() - t_wrap0) * 1000.0)

    req_ids = getattr(ctx, "current_batch_req_ids", None)
    if not req_ids: # Empty or None
        out.zero_()
        return
        
    kv_dtype = ctx.kv_cache.dtype
    
    def _run():
        t_cast0 = time.perf_counter() if prof_detail else 0.0
        q_in = q if q.dtype == kv_dtype else q.to(kv_dtype)
        k_in = k if k.dtype == kv_dtype else k.to(kv_dtype)
        v_in = v if v.dtype == kv_dtype else v.to(kv_dtype)
        if prof_detail and prof is not None:
            prof["attn_cast_ms"] = float(prof.get("attn_cast_ms", 0.0)) + float((time.perf_counter() - t_cast0) * 1000.0)

        t_pe0 = time.perf_counter() if prof_detail else 0.0
        pe = getattr(ctx, "_pe_name", None)
        if pe is None or int(getattr(ctx, "_pe_code", -1)) != int(pos_encoding_mode):
            if isinstance(pos_encoding_mode, int):
                from flashinfer.utils import PosEncodingMode
                try:
                    pe = PosEncodingMode(pos_encoding_mode).name
                except ValueError:
                    pe = pos_encoding_mode
            else:
                pe = pos_encoding_mode
            setattr(ctx, "_pe_name", pe)
            try:
                setattr(ctx, "_pe_code", int(pos_encoding_mode))
            except Exception:
                setattr(ctx, "_pe_code", -1)
        if prof_detail and prof is not None:
            prof["attn_pe_ms"] = float(prof.get("attn_pe_ms", 0.0)) + float((time.perf_counter() - t_pe0) * 1000.0)

        out_temp = out
        if out.dtype != kv_dtype:
            out_temp = torch.empty((int(total_tokens), int(num_heads), int(head_dim)), device=device, dtype=kv_dtype)
        t_run0 = time.perf_counter() if prof_detail else 0.0
        run_attention_kernel(q_in, k_in, v_in, out_temp, ctx, int(layer_idx), float(rope_theta), str(pe))
        if prof_detail and prof is not None:
            prof["attn_run_ms"] = float(prof.get("attn_run_ms", 0.0)) + float((time.perf_counter() - t_run0) * 1000.0)
        if out_temp is not out:
            t_outcast0 = time.perf_counter() if prof_detail else 0.0
            out.copy_(out_temp.to(out.dtype))
            if prof_detail and prof is not None:
                prof["attn_outcast_ms"] = float(prof.get("attn_outcast_ms", 0.0)) + float((time.perf_counter() - t_outcast0) * 1000.0)
        if layer_idx == 0 and bool(int(os.environ.get("LLMSCHEDULER_CHECK_ATTENTION_NAN", "0"))):
            t_nan0 = time.perf_counter() if prof_detail else 0.0
            if torch.isnan(out).any():
                out.zero_()
            if prof_detail and prof is not None:
                prof["attn_nan_ms"] = float(prof.get("attn_nan_ms", 0.0)) + float((time.perf_counter() - t_nan0) * 1000.0)

    ext = getattr(ctx, "_trt_external_stream", None)
    if ext is not None:
        with torch.cuda.stream(ext):
            _run()
    else:
        _run()

    if prof_on and prof is not None:
        prof["attn_impl_ms"] = float(prof.get("attn_impl_ms", 0.0)) + float((time.perf_counter() - t_attn0) * 1000.0)


def run_attention_kernel(q, k, v, out, ctx, layer_idx, rope_theta, pos_encoding_mode):
    prof = getattr(ctx, "_prof", None)
    prof_on = (prof is not None) and bool(int(os.environ.get("LLMSCHEDULER_PROFILE", "0")))
    prof_events = getattr(ctx, "_prof_events", None) if prof_on else None
    t0 = time.perf_counter() if prof_on else 0.0

    step_id = int(getattr(ctx, "_engine_step_id", 0) or 0)
    cache = getattr(ctx, "_step_cache", None)
    cached_step_id = int(cache.get("step_id", -1)) if isinstance(cache, dict) else -1
    cached_sig = cache.get("sig") if isinstance(cache, dict) else None
    sig = (
        tuple(int(x) for x in ctx.current_batch_req_ids),
        tuple(int(x) for x in ctx.current_batch_total_lens),
        tuple(int(x) for x in ctx.current_batch_seq_lens),
        tuple(bool(x) for x in getattr(ctx, "current_batch_is_prefill", [])),
    )
    if (not isinstance(cache, dict)) or cached_step_id != step_id or cached_sig != sig:
        meta = ctx.mb.prepare_step(
            ctx.current_batch_req_ids,
            ctx.current_batch_total_lens,
            ctx.current_batch_seq_lens,
            ctx.page_size,
            int(layer_idx),
            ctx.num_layers,
        )
        ctx._step_cache = {
            "step_id": int(step_id),
            "sig": sig,
            "meta": meta,
            "seq_lens": list(ctx.current_batch_seq_lens),
        }

        batch_size = int(len(ctx.current_batch_req_ids))
        seq_lens_cpu = list(ctx.current_batch_seq_lens)
        if len(seq_lens_cpu) != batch_size:
            raise RuntimeError(f"Invalid ctx.current_batch_seq_lens len={len(seq_lens_cpu)} batch={batch_size}")
        is_prefill_flags = list(getattr(ctx, "current_batch_is_prefill", []))
        if len(is_prefill_flags) != batch_size:
            is_prefill_flags = [False for _ in range(batch_size)]
        prefill_reqs = 0
        for f in is_prefill_flags:
            if bool(f):
                prefill_reqs += 1
            else:
                break
        prefill_tokens = int(sum(int(x) for x in seq_lens_cpu[:prefill_reqs]))
        decode_reqs = int(batch_size - prefill_reqs)
        ctx._step_cache["prefill_reqs"] = int(prefill_reqs)
        ctx._step_cache["prefill_tokens"] = int(prefill_tokens)
        ctx._step_cache["decode_reqs"] = int(decode_reqs)

        indptr = meta.indptr.contiguous()
        indices = meta.indices.contiguous()
        last_page_len = meta.last_page_len.contiguous()
        qo_indptr = meta.qo_indptr.contiguous()
        slot_mapping = meta.slot_mapping.contiguous()
        ctx._step_cache["indptr"] = indptr
        ctx._step_cache["indices"] = indices
        ctx._step_cache["last_page_len"] = last_page_len
        ctx._step_cache["qo_indptr"] = qo_indptr
        ctx._step_cache["slot_mapping"] = slot_mapping

        if int(prefill_reqs) > 0:
            qo_indptr_prefill = qo_indptr[: int(prefill_reqs) + 1].contiguous()
            indptr_prefill = indptr[: int(prefill_reqs) + 1].contiguous()
            indices_prefill = indices[: int(indptr_prefill[-1].item())].contiguous()
            last_page_len_prefill = last_page_len[: int(prefill_reqs)].contiguous()

            history_lens = list(getattr(ctx, "current_batch_history_lens", []))
            can_use_ragged = bool(decode_reqs == 0) and bool(len(history_lens) == batch_size) and all(
                int(x) == 0 for x in history_lens[: int(prefill_reqs)]
            )
            disable_ragged = bool(int(os.environ.get("LLMSCHEDULER_DISABLE_RAGGED_PREFILL", "0")))
            if disable_ragged:
                can_use_ragged = False
            ctx._step_cache["use_ragged_prefill"] = bool(can_use_ragged)

        ctx._step_cache["append_paged_kv_cache"] = flashinfer.page.append_paged_kv_cache
        plan_wrappers(
            ctx,
            meta,
            q.dtype,
            float(rope_theta),
            str(pos_encoding_mode),
            int(prefill_reqs),
            int(decode_reqs),
        )

    indptr = ctx._step_cache["indptr"]
    indices = ctx._step_cache["indices"]
    last_page_len = ctx._step_cache["last_page_len"]
    qo_indptr = ctx._step_cache["qo_indptr"]
    slot_mapping = ctx._step_cache["slot_mapping"]

    if prof_on:
        prof["meta_ms"] = float(prof.get("meta_ms", 0.0)) + float((time.perf_counter() - t0) * 1000.0)

    capturing = bool(int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH", "0")))
    if not capturing:
        try:
            capturing = bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            capturing = False
    if not capturing:
        if int(qo_indptr[-1].item()) != int(q.shape[0]) or int(slot_mapping.numel()) != int(q.shape[0]):
            raise RuntimeError(
                f"Invalid metadata: qo_indptr[-1]={int(qo_indptr[-1].item())}, slot_mapping={int(slot_mapping.numel())}, total_tokens={int(q.shape[0])}"
            )
        if int(indptr[-1].item()) != int(indices.numel()):
            raise RuntimeError(f"Invalid metadata: indptr[-1]={int(indptr[-1].item())}, indices={int(indices.numel())}")
    
    q_use = q
    k_use = k
    pos_encoding_mode_use = pos_encoding_mode
    sm_scale = 1.0 / (float(ctx.head_dim) ** 0.5)

    prefill_reqs = int(ctx._step_cache.get("prefill_reqs", 0))
    prefill_tokens = int(ctx._step_cache.get("prefill_tokens", 0))
    decode_reqs = int(ctx._step_cache.get("decode_reqs", 0))
    use_ragged_prefill = bool(ctx._step_cache.get("use_ragged_prefill", False))

    meta = ctx._step_cache["meta"]
    paged_kv_cache = ctx.kv_cache[int(layer_idx)]

    t1 = time.perf_counter() if prof_on else 0.0
    if prof_on:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
    kv_append(ctx, meta, k_use, v, paged_kv_cache)
    if prof_on:
        e1.record()
        if isinstance(prof_events, dict):
            prof_events.setdefault("kv_write", []).append((e0, e1))
        prof["kv_write_ms"] = float(prof.get("kv_write_ms", 0.0)) + float((time.perf_counter() - t1) * 1000.0)

    if int(prefill_tokens) > 0:
        t2 = time.perf_counter() if prof_on else 0.0
        if prof_on:
            f0 = torch.cuda.Event(enable_timing=True)
            f1 = torch.cuda.Event(enable_timing=True)
            f0.record()
        run_prefill(ctx, q_use, k_use, v, out, paged_kv_cache, int(prefill_tokens))
        if prof_on:
            f1.record()
            if isinstance(prof_events, dict):
                prof_events.setdefault("flashinfer", []).append((f0, f1))
            prof["flashinfer_prefill_ms"] = float(prof.get("flashinfer_prefill_ms", 0.0)) + float((time.perf_counter() - t2) * 1000.0)

    if int(decode_reqs) > 0:
        t2 = time.perf_counter() if prof_on else 0.0
        if prof_on:
            f0 = torch.cuda.Event(enable_timing=True)
            f1 = torch.cuda.Event(enable_timing=True)
            f0.record()
        run_decode(ctx, q_use, out, paged_kv_cache, int(prefill_tokens), int(decode_reqs))
        if prof_on:
            f1.record()
            if isinstance(prof_events, dict):
                prof_events.setdefault("flashinfer", []).append((f0, f1))
            prof["flashinfer_decode_ms"] = float(prof.get("flashinfer_decode_ms", 0.0)) + float((time.perf_counter() - t2) * 1000.0)
    return
