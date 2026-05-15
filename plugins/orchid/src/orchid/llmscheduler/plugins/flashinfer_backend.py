import flashinfer
import torch
import os


def plan_wrappers(ctx, meta, q_dtype, rope_theta, pos_encoding_mode, prefill_reqs, decode_reqs):
    indptr = meta.indptr.contiguous()
    indices = meta.indices.contiguous()
    last_page_len = meta.last_page_len.contiguous()
    qo_indptr = meta.qo_indptr.contiguous()

    if int(prefill_reqs) > 0:
        qo_indptr_prefill = qo_indptr[: int(prefill_reqs) + 1].contiguous()
        indptr_prefill = indptr[: int(prefill_reqs) + 1].contiguous()
        indices_prefill = indices[: int(indptr_prefill[-1].item())].contiguous()
        last_page_len_prefill = last_page_len[: int(prefill_reqs)].contiguous()

        if bool(ctx._step_cache.get("use_ragged_prefill", False)):
            ctx.prefill_ragged_wrapper.plan(
                qo_indptr_prefill,
                qo_indptr_prefill,
                int(ctx.num_heads),
                int(ctx.kv_num_heads),
                int(ctx.head_dim),
                causal=True,
                pos_encoding_mode=str(pos_encoding_mode),
                sm_scale=1.0 / (float(ctx.head_dim) ** 0.5),
                rope_theta=float(rope_theta),
                q_data_type=q_dtype,
                kv_data_type=q_dtype,
                o_data_type=q_dtype,
            )
        else:
            ctx.prefill_paged_wrapper.plan(
                qo_indptr_prefill,
                indptr_prefill,
                indices_prefill,
                last_page_len_prefill,
                int(ctx.num_heads),
                int(ctx.kv_num_heads),
                int(ctx.head_dim),
                int(ctx.page_size),
                causal=True,
                pos_encoding_mode=str(pos_encoding_mode),
                sm_scale=1.0 / (float(ctx.head_dim) ** 0.5),
                rope_theta=float(rope_theta),
                q_data_type=q_dtype,
                kv_data_type=q_dtype,
                o_data_type=q_dtype,
            )

    if int(decode_reqs) > 0:
        base_pages = int(indptr[int(prefill_reqs)].item())
        decode_indptr = (indptr[int(prefill_reqs) :] - int(base_pages)).contiguous()
        decode_indices = indices[int(base_pages) :].contiguous()
        decode_last_page_len = last_page_len[int(prefill_reqs) :].contiguous()
        disable_split_kv = bool(int(os.environ.get("LLMSCHEDULER_FLASHINFER_DISABLE_SPLIT_KV", "0")))
        if bool(int(os.environ.get("LLMSCHEDULER_FLASHINFER_DECODE_CUDAGRAPH", "0"))) and hasattr(ctx, "_decode_cudagraph_buffers"):
            bufs = ctx._decode_cudagraph_buffers
            max_bs = int(bufs.get("max_bs") or 0)
            if int(decode_reqs) > int(max_bs):
                raise RuntimeError(f"decode_reqs {int(decode_reqs)} exceeds max_bs {int(max_bs)} for cudagraph")
            indptr_buf = bufs["indptr"]
            last_buf = bufs["last_page_len"]
            indices_buf = bufs["indices"]
            indptr_buf[: int(decode_reqs) + 1].copy_(decode_indptr)
            indptr_buf[int(decode_reqs) + 1 : int(max_bs) + 1].fill_(int(indptr_buf[int(decode_reqs)].item()))
            last_buf[: int(decode_reqs)].copy_(decode_last_page_len)
            if int(max_bs) > int(decode_reqs):
                last_buf[int(decode_reqs) : int(max_bs)].fill_(1)
            indices_buf[: int(decode_indices.numel())].copy_(decode_indices)
            ctx.decode_wrapper.plan(
                indptr_buf[: int(max_bs) + 1],
                indices_buf,
                last_buf[: int(max_bs)],
                int(ctx.num_heads),
                int(ctx.kv_num_heads),
                int(ctx.head_dim),
                int(ctx.page_size),
                pos_encoding_mode=str(pos_encoding_mode),
                sm_scale=1.0 / (float(ctx.head_dim) ** 0.5),
                rope_theta=float(rope_theta),
                q_data_type=q_dtype,
                kv_data_type=q_dtype,
                o_data_type=q_dtype,
                disable_split_kv=bool(disable_split_kv),
            )
        else:
            ctx.decode_wrapper.plan(
                decode_indptr,
                decode_indices,
                decode_last_page_len,
                int(ctx.num_heads),
                int(ctx.kv_num_heads),
                int(ctx.head_dim),
                int(ctx.page_size),
                pos_encoding_mode=str(pos_encoding_mode),
                sm_scale=1.0 / (float(ctx.head_dim) ** 0.5),
                rope_theta=float(rope_theta),
                q_data_type=q_dtype,
                kv_data_type=q_dtype,
                o_data_type=q_dtype,
                disable_split_kv=bool(disable_split_kv),
            )


def kv_append(ctx, meta, k_use, v_use, paged_kv_cache):
    append_paged_kv_cache = ctx._step_cache["append_paged_kv_cache"]
    indptr = ctx._step_cache["indptr"]
    indices = ctx._step_cache["indices"]
    last_page_len = ctx._step_cache["last_page_len"]
    append_paged_kv_cache(
        k_use,
        v_use,
        meta.batch_indices,
        meta.positions,
        paged_kv_cache,
        indices,
        indptr,
        last_page_len,
        kv_layout="NHD",
    )


def run_prefill(ctx, q_use, k_use, v_use, out, paged_kv_cache, prefill_tokens):
    if int(prefill_tokens) <= 0:
        return
    q_prefill = q_use[: int(prefill_tokens)]
    k_prefill = k_use[: int(prefill_tokens)]
    v_prefill = v_use[: int(prefill_tokens)]
    out_prefill = out[: int(prefill_tokens)]
    if bool(ctx._step_cache.get("use_ragged_prefill", False)):
        ctx.prefill_ragged_wrapper.run(q_prefill, k_prefill, v_prefill, out=out_prefill)
    else:
        ctx.prefill_paged_wrapper.run(q_prefill, paged_kv_cache, out=out_prefill)


def run_decode(ctx, q_use, out, paged_kv_cache, prefill_tokens, decode_reqs):
    if int(decode_reqs) <= 0:
        return
    q_decode_tokens = q_use[int(prefill_tokens) :]
    out_decode_tokens = out[int(prefill_tokens) :]
    if int(q_decode_tokens.shape[0]) != int(decode_reqs):
        raise RuntimeError(f"decode expects tokens==decode_reqs, got {int(q_decode_tokens.shape[0])} vs {int(decode_reqs)}")
    q_decode = q_decode_tokens.view(int(decode_reqs), int(ctx.num_heads), int(ctx.head_dim))
    out_decode = out_decode_tokens.view(int(decode_reqs), int(ctx.num_heads), int(ctx.head_dim))
    if bool(int(os.environ.get("LLMSCHEDULER_FLASHINFER_DECODE_CUDAGRAPH", "0"))) and hasattr(ctx, "_decode_cudagraph_buffers"):
        max_bs = int(ctx._decode_cudagraph_buffers.get("max_bs") or 0)
        if int(decode_reqs) > int(max_bs):
            raise RuntimeError(f"decode_reqs {int(decode_reqs)} exceeds max_bs {int(max_bs)} for cudagraph")
        q_pad = ctx._step_cache.get("decode_q_pad")
        out_pad = ctx._step_cache.get("decode_out_pad")
        if (
            q_pad is None
            or out_pad is None
            or int(q_pad.shape[0]) != int(max_bs)
            or int(out_pad.shape[0]) != int(max_bs)
            or q_pad.dtype != q_decode.dtype
            or out_pad.dtype != out_decode.dtype
            or q_pad.device != q_decode.device
        ):
            q_pad = torch.empty((int(max_bs), int(ctx.num_heads), int(ctx.head_dim)), dtype=q_decode.dtype, device=q_decode.device)
            out_pad = torch.empty((int(max_bs), int(ctx.num_heads), int(ctx.head_dim)), dtype=out_decode.dtype, device=out_decode.device)
            ctx._step_cache["decode_q_pad"] = q_pad
            ctx._step_cache["decode_out_pad"] = out_pad
        q_pad[: int(decode_reqs)].copy_(q_decode)
        if int(max_bs) > int(decode_reqs):
            q_pad[int(decode_reqs) :].zero_()
        ctx.decode_wrapper.run(q_pad, paged_kv_cache, out=out_pad)
        out_decode.copy_(out_pad[: int(decode_reqs)])
    else:
        ctx.decode_wrapper.run(q_decode, paged_kv_cache, out=out_decode)
