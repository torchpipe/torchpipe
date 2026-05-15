import torch


def test_pause_req_frees_pages():
    from orchid.llmscheduler.core.engine import ContinuousBatchingEngine

    class _PM:
        def __init__(self):
            self.freed = []

        def free(self, rid: int):
            self.freed.append(int(rid))

    class _Ctx:
        def __init__(self):
            self.page_manager = _PM()

    e = ContinuousBatchingEngine.__new__(ContinuousBatchingEngine)
    e.requests = {
        1: {
            "prompt_ids": torch.as_tensor([1, 2, 3], dtype=torch.int64, device="cpu"),
            "prompt_full_len": 3,
            "generated": [4, 5],
            "emitted_len": 2,
            "history_len": 5,
            "pages_allocated": 3,
            "pages_cap": 7,
            "prefill_cursor": 0,
            "is_prefill": False,
            "finished": False,
        }
    }
    e.waiting_queue = []
    e.running_queue = [1]
    e.used_pages = 3
    e.reserved_pages = 7
    e.max_pages = 64
    e.page_size = 16
    e._log_engine = False

    ctx = _Ctx()
    ctx.num_layers = 1
    ok = e._pause_req(ctx, 1)
    assert int(ok) == 3
    assert ctx.page_manager.freed == [1]
    assert 1 not in e.running_queue
    assert 1 in e.waiting_queue
    assert int(e.used_pages) == 0
    assert int(e.reserved_pages) == 7
