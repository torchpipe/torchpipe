from __future__ import annotations

import asyncio
import threading
import queue as pyqueue
from typing import Any, AsyncIterator, Optional

import janus

from ..core.engine import ContinuousBatchingEngine
from ..runtime.base import AttentionContext


class OfflineEngine:
    def __init__(self, engine: ContinuousBatchingEngine, ctx: AttentionContext):
        self.engine = engine
        self.ctx = ctx
        self.request_queue: janus.Queue = janus.Queue()
        self.results_queues: dict[int, tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = {}
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._next_req_id = 1
        self._consecutive_step_failures = 0

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join()

    def _alloc_req_id(self) -> int:
        rid = int(self._next_req_id & 0x7FFFFFFF)
        self._next_req_id += 1
        if rid == 0:
            rid = int(self._next_req_id & 0x7FFFFFFF)
            self._next_req_id += 1
        return rid
    
    def _cleanup_failed_reqs(self, req_ids: list[int], err: str) -> None:
        for req_id in req_ids:
            item = self.results_queues.get(int(req_id))
            if item is None:
                continue
            q, loop = item
            loop.call_soon_threadsafe(q.put_nowait, {"req_id": int(req_id), "error": str(err), "finished": True})
            loop.call_soon_threadsafe(q.put_nowait, None)
            try:
                del self.results_queues[int(req_id)]
            except Exception:
                pass

    def _fail_req(self, req_id: int, q: asyncio.Queue, loop: asyncio.AbstractEventLoop, err: str) -> None:
        loop.call_soon_threadsafe(q.put_nowait, {"req_id": int(req_id), "error": str(err), "finished": True})
        loop.call_soon_threadsafe(q.put_nowait, None)

    def _reset_engine_state(self) -> None:
        try:
            self.engine.requests.clear()
        except Exception:
            pass
        try:
            self.engine.running_queue.clear()
        except Exception:
            pass
        try:
            self.engine.waiting_queue.clear()
        except Exception:
            pass
        try:
            self.engine.used_pages = 0
        except Exception:
            pass
        try:
            pm = getattr(self.ctx, "page_manager", None)
            if pm is not None and hasattr(pm, "reset"):
                pm.reset()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _run_loop(self) -> None:
        while self.running:
            drained = 0
            drain_max = 256
            try:
                import os
                drain_max = int(os.environ.get("LLMSCHEDULER_DRAIN_MAX", "256"))
            except Exception:
                drain_max = 256
            drain_max = int(max(1, drain_max))
            for _ in range(drain_max):
                try:
                    req_data = self.request_queue.sync_q.get_nowait()
                except janus.SyncQueueEmpty:
                    break
                req_id = int(req_data["req_id"])
                input_ids = req_data["input_ids"]
                max_tokens = int(req_data["max_tokens"])
                out_q = req_data["queue"]
                out_loop = req_data["loop"]
                want_text = bool(req_data.get("want_text", True))
                try:
                    self.engine.add_request(req_id, input_ids, max_tokens, want_text=want_text)
                    self.results_queues[req_id] = (out_q, out_loop)
                except Exception as e:
                    self._fail_req(req_id, out_q, out_loop, str(e))
                drained += 1

            if not self.engine.running_queue and drained == 0:
                try:
                    req_data = self.request_queue.sync_q.get(timeout=0.05)
                    req_id = int(req_data["req_id"])
                    input_ids = req_data["input_ids"]
                    max_tokens = int(req_data["max_tokens"])
                    out_q = req_data["queue"]
                    out_loop = req_data["loop"]
                    want_text = bool(req_data.get("want_text", True))
                    try:
                        self.engine.add_request(req_id, input_ids, max_tokens, want_text=want_text)
                        self.results_queues[req_id] = (out_q, out_loop)
                    except Exception as e:
                        self._fail_req(req_id, out_q, out_loop, str(e))
                except (pyqueue.Empty, janus.SyncQueueEmpty):
                    pass

            if not self.engine.running_queue and not self.engine.waiting_queue:
                continue

            try:
                step_outputs = self.engine.step(self.ctx)
            except Exception as e:
                err = str(e)
                self._consecutive_step_failures += 1
                try:
                    pause_req = getattr(self.engine, "_pause_req", None)
                    if callable(pause_req):
                        for rid in list(self.engine.running_queue):
                            try:
                                pause_req(self.ctx, int(rid))
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    self.ctx.current_batch_req_ids = []
                    self.ctx.current_batch_seq_lens = []
                    self.ctx.current_batch_total_lens = []
                except Exception:
                    pass
                if self._consecutive_step_failures >= 3:
                    try:
                        req_ids = [int(x) for x in list(self.results_queues.keys())]
                    except Exception:
                        req_ids = []
                    self._cleanup_failed_reqs(req_ids, err)
                    self._reset_engine_state()
                    self._consecutive_step_failures = 0
                continue
            self._consecutive_step_failures = 0
            if step_outputs:
                for out in step_outputs:
                    req_id = int(out["req_id"])
                    item = self.results_queues.get(req_id)
                    if item is None:
                        continue
                    q, loop = item
                    loop.call_soon_threadsafe(q.put_nowait, out)
                    if out["finished"]:
                        loop.call_soon_threadsafe(q.put_nowait, None)
                        del self.results_queues[req_id]

    async def add_request(self, req_id: int, input_ids: Any, max_tokens: int, *, want_text: bool = True) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        await self.request_queue.async_q.put(
            {
                "req_id": int(req_id),
                "input_ids": input_ids,
                "max_tokens": int(max_tokens),
                "queue": q,
                "loop": loop,
                "want_text": bool(want_text),
            }
        )
        return q

    async def submit(self, input_ids: Any, max_tokens: int, *, want_text: bool = True) -> asyncio.Queue:
        return await self.add_request(self._alloc_req_id(), input_ids, max_tokens, want_text=want_text)

    async def generate(self, input_ids: Any, max_tokens: int) -> str:
        q = await self.submit(input_ids, max_tokens)
        out = ""
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, dict) and item.get("error"):
                raise RuntimeError(str(item.get("error")))
            out += str(item["text"])
        return out

    async def stream(self, input_ids: Any, max_tokens: int) -> AsyncIterator[dict[str, Any]]:
        q = await self.submit(input_ids, max_tokens)
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, dict) and item.get("error"):
                raise RuntimeError(str(item.get("error")))
            yield item


class TensorRTOfflineEngine:
    def __init__(
        self,
        model_path: str,
        tokenizer,
        *,
        use_fp16: bool = True,
        engine_path: str | None = None,
        max_pages: int = 4096,
        page_size: int = 16,
        device: str = "cuda",
        max_batch_tokens: int | None = None,
        use_cpp_metadata: bool = True,
        num_layers: int | None = None,
        num_heads: int | None = None,
        kv_num_heads: int | None = None,
        head_dim: int | None = None,
    ):
        from ..model_params import infer_model_params
        from ..runtime.trt_runtime import TensorRTModelRuntime

        runtime = TensorRTModelRuntime(model_path, use_fp16=use_fp16, engine_path=engine_path)
        core_engine = ContinuousBatchingEngine(runtime, tokenizer, max_pages=max_pages, page_size=page_size, device=device)

        p = infer_model_params(model_path, getattr(tokenizer, "name_or_path", None) or None)
        nl = int(num_layers) if num_layers is not None else int(p.num_layers)
        nh = int(num_heads) if num_heads is not None else int(p.num_heads)
        nkh = int(kv_num_heads) if kv_num_heads is not None else int(p.kv_num_heads)
        hd = int(head_dim) if head_dim is not None else int(p.head_dim)
        ctx = AttentionContext(
            num_layers=nl,
            num_heads=nh,
            kv_num_heads=nkh,
            head_dim=hd,
            page_size=page_size,
            max_pages=max_pages,
            use_cpp_metadata=use_cpp_metadata,
            device=device,
            use_fp16=use_fp16,
        )
        core_engine.max_pages = int(ctx.pages_per_layer)
        if max_batch_tokens is not None:
            core_engine.max_batch_tokens = int(max_batch_tokens)

        self.tokenizer = tokenizer
        self.runtime = runtime
        self.engine = core_engine
        self.ctx = ctx
        self._runner = OfflineEngine(core_engine, ctx)
        self._runner.start()

    def close(self):
        self._runner.stop()
        try:
            self.ctx.close()
        except Exception:
            pass

    async def add_request(self, req_id: int, input_ids: Any, max_tokens: int, *, want_text: bool = True) -> asyncio.Queue:
        return await self._runner.add_request(req_id, input_ids, max_tokens, want_text=want_text)

    async def submit(self, input_ids: Any, max_tokens: int, *, want_text: bool = True) -> asyncio.Queue:
        return await self._runner.submit(input_ids, max_tokens, want_text=want_text)

    async def generate(self, input_ids: Any, max_tokens: int) -> str:
        return await self._runner.generate(input_ids, max_tokens)

    async def stream(self, input_ids: Any, max_tokens: int) -> AsyncIterator[dict[str, Any]]:
        async for item in self._runner.stream(input_ids, max_tokens):
            yield item
