import torch


def test_schedule_step_plan_exists_and_runs():
    import tvm_ffi
    from orchid.llmscheduler.core.allocator import ensure_scheduler_plan_ops_loaded

    ensure_scheduler_plan_ops_loaded()
    f = tvm_ffi.get_global_func("custom.schedule_step_plan", allow_missing=True)
    pack = tvm_ffi.get_global_func("custom.pack_padded_prefix_i64", allow_missing=True)
    assert f is not None
    assert pack is not None

    running_req_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    is_prefill_flags = torch.tensor([1, 0, 1], dtype=torch.int32)
    prefill_remaining = torch.tensor([10, 0, 5], dtype=torch.int32)
    decode_ready = torch.tensor([0, 1, 0], dtype=torch.int32)

    ret = f(
        running_req_ids,
        is_prefill_flags,
        prefill_remaining,
        decode_ready,
        16,
        8,
        0.5,
        2,
        0,
    )
    assert len(ret) >= 4
    out_ids = torch.from_dlpack(ret[0].__dlpack__())
    out_nt = torch.from_dlpack(ret[1].__dlpack__())
    out_pf = torch.from_dlpack(ret[2].__dlpack__())
    out_so = torch.from_dlpack(ret[3].__dlpack__())
    assert out_ids.dtype == torch.int32
    assert out_nt.dtype == torch.int32
    assert out_pf.dtype == torch.int32
    assert out_so.dtype == torch.int32
    assert out_ids.numel() == out_nt.numel() == out_pf.numel() == out_so.numel()

    padded = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.int64)
    lengths = torch.tensor([3, 2], dtype=torch.int32)
    flat = pack(padded, lengths)
    out = torch.from_dlpack(flat.__dlpack__()).to("cpu")
    assert out.dtype == torch.int64
    assert out.tolist() == [1, 2, 3, 4, 5]
