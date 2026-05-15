import os
import time
import torch
import numpy as np
import tvm_ffi
from ..runtime.base import ModelRuntime, AttentionContext

_PY_SCHED_STEP_REGISTERED = False


def _ensure_schedule_step_registered() -> None:
    global _PY_SCHED_STEP_REGISTERED
    if _PY_SCHED_STEP_REGISTERED:
        return
    f = tvm_ffi.get_global_func("custom.schedule_step", allow_missing=True)
    if f is not None:
        _PY_SCHED_STEP_REGISTERED = True
        return

    @tvm_ffi.register_global_func("custom.schedule_step")
    def _py_schedule_step(
        running_req_ids,
        is_prefill_flags,
        prefill_remaining,
        decode_ready,
        max_batch_tokens,
        prefill_max_chunk,
        prefill_budget_fraction,
        decode_reserve_reqs,
        strict_no_mix,
    ):
        rids = torch.as_tensor(running_req_ids, dtype=torch.int32, device="cpu").view(-1).tolist()
        is_pf = torch.as_tensor(is_prefill_flags, dtype=torch.int32, device="cpu").view(-1).tolist()
        pref_rem = torch.as_tensor(prefill_remaining, dtype=torch.int32, device="cpu").view(-1).tolist()
        dec_ready = torch.as_tensor(decode_ready, dtype=torch.int32, device="cpu").view(-1).tolist()

        cap_tokens = int(max_batch_tokens)
        prefill_cap = int(max(1, int(prefill_max_chunk)))
        frac = float(prefill_budget_fraction)
        frac = float(max(0.0, min(1.0, frac)))
        prefill_budget_limit = int(float(cap_tokens) * frac)
        reserve_decode = int(max(0, int(decode_reserve_reqs)))
        no_mix = bool(int(strict_no_mix))

        decode_candidates = sum(1 for i in range(len(rids)) if (not bool(is_pf[i])) and bool(dec_ready[i]))
        reserve = int(min(reserve_decode, decode_candidates))

        sel_ids = []
        sel_nt = []
        sel_pf = []
        sel_so = []

        used_tokens = 0
        used_prefill_tokens = 0
        has_prefill = False

        for req_id, pf, rem, dr in zip(rids, is_pf, pref_rem, dec_ready):
            if bool(pf):
                rem = int(rem)
                if rem <= 0:
                    continue
                budget_left = int(cap_tokens) - int(used_tokens)
                if budget_left <= 0:
                    continue
                if not no_mix:
                    budget_left = max(0, int(budget_left) - int(reserve))
                    prefill_budget_left = int(prefill_budget_limit) - int(used_prefill_tokens)
                    budget_left = min(int(budget_left), max(0, int(prefill_budget_left)))
                if no_mix and rem > budget_left:
                    continue
                chunk = rem if no_mix else min(rem, budget_left, prefill_cap)
                if chunk <= 0:
                    continue
                if used_tokens + chunk > cap_tokens:
                    continue
                sel_ids.append(int(req_id))
                sel_nt.append(int(chunk))
                sel_pf.append(1)
                sel_so.append(1 if (no_mix or chunk >= rem) else 0)
                used_tokens += int(chunk)
                used_prefill_tokens += int(chunk)
                has_prefill = True
                continue

            if no_mix and has_prefill:
                continue
            if not bool(dr):
                continue
            if used_tokens + 1 > cap_tokens:
                continue
            sel_ids.append(int(req_id))
            sel_nt.append(1)
            sel_pf.append(0)
            sel_so.append(1)
            used_tokens += 1

        return (
            torch.as_tensor(sel_ids, dtype=torch.int32, device="cpu"),
            torch.as_tensor(sel_nt, dtype=torch.int32, device="cpu"),
            torch.as_tensor(sel_pf, dtype=torch.int32, device="cpu"),
            torch.as_tensor(sel_so, dtype=torch.int32, device="cpu"),
        )

    _PY_SCHED_STEP_REGISTERED = True

class ContinuousBatchingEngine:
    def __init__(self, runtime: ModelRuntime, tokenizer, max_pages=4096, page_size=16, device: str = "cuda"):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.max_pages = max_pages
        self.page_size = page_size
        self.device = str(device)
        
        self.requests = {} 
        self.running_queue = [] 
        self.waiting_queue = [] 
        self.used_pages = 0
        
        self.last_step_logits = None
        self._log_engine = bool(os.environ.get("LLMSCHEDULER_LOG_ENGINE"))
        try:
            self.max_batch_tokens = int(os.environ.get("LLMSCHEDULER_MAX_BATCH_TOKENS", "4096"))
        except Exception:
            self.max_batch_tokens = 4096
        try:
            self._log_engine_interval_s = float(os.environ.get("LLMSCHEDULER_LOG_ENGINE_INTERVAL_S", "1.0"))
        except Exception:
            self._log_engine_interval_s = 1.0
        self._last_engine_log_t = 0.0
        _ensure_schedule_step_registered()
        self._prof_step = 0
        self._prof_print_interval = int(os.environ.get("LLMSCHEDULER_PROFILE_INTERVAL", "20"))
    
    def _ceil_div(self, a: int, b: int) -> int:
        return (int(a) + int(b) - 1) // int(b)
    
    def _as_np_int64(self, x) -> np.ndarray:
        if isinstance(x, np.ndarray):
            if x.dtype == np.int64:
                return x
            return x.astype(np.int64, copy=False)
        return np.array(x, dtype=np.int64)
    
    def _cap_tokens_per_req(self, ctx: AttentionContext) -> int:
        pages_budget = int(getattr(ctx, "pages_per_layer", 0) or 0)
        if pages_budget <= 0:
            num_layers = int(getattr(ctx, "num_layers", 1) or 1)
            pages_budget = int(self.max_pages) // max(1, num_layers)
        pages_budget = int(max(0, pages_budget))
        if pages_budget <= 0:
            return 0
        return int(pages_budget) * int(self.page_size)
    
    def _truncate_to_tokens(self, ids: np.ndarray, cap_tokens: int) -> tuple[np.ndarray, int]:
        cap_tokens = int(cap_tokens)
        if cap_tokens <= 0:
            return ids[-0:], int(len(ids))
        if len(ids) <= cap_tokens:
            return ids, 0
        dropped = int(len(ids) - cap_tokens)
        return ids[-cap_tokens:], dropped
    
    def _pages_for_running_req(self, ctx: AttentionContext, req: dict) -> int:
        if "pages_allocated" in req:
            return int(req.get("pages_allocated", 0) or 0)
        if int(req.get("history_len", 0)) > 0:
            pages_per_layer = self._ceil_div(int(req["history_len"]), int(self.page_size))
        else:
            pages_per_layer = self._ceil_div(int(len(req["input_ids"])), int(self.page_size))
        return int(pages_per_layer)
    
    def _free_kv(self, ctx: AttentionContext, req_id: int) -> None:
        rid = int(req_id)
        ctx.page_manager.free(int(rid))
    
    def _pause_req(self, ctx: AttentionContext, req_id: int) -> int:
        req_id = int(req_id)
        req = self.requests.get(req_id)
        if req is None or bool(req.get("finished")):
            return 0
        
        pages_used = int(self._pages_for_running_req(ctx, req))
        self.used_pages = int(max(0, int(self.used_pages) - pages_used))
        
        if req_id in self.running_queue:
            try:
                self.running_queue.remove(req_id)
            except Exception:
                pass
        
        prompt_ids = self._as_np_int64(req.get("prompt_ids", req.get("input_ids")))
        gen = req.get("generated") or []
        gen_ids = self._as_np_int64(gen)
        merged = np.concatenate([prompt_ids, gen_ids]) if len(gen_ids) else prompt_ids
        merged, dropped = self._truncate_to_tokens(merged, self._cap_tokens_per_req(ctx))
        
        req["input_ids"] = merged
        req["history_len"] = 0
        req["pages_allocated"] = 0
        req["is_prefill"] = True
        req["prefill_cursor"] = 0
        if dropped:
            req["prefix_truncated"] = True
            req["prefix_dropped_tokens"] = int(req.get("prefix_dropped_tokens", 0)) + int(dropped)
        
        self._free_kv(ctx, req_id)
        self.waiting_queue.append(req_id)
        if self._log_engine:
            print(
                f"[engine] pause req_id={req_id} freed_pages={pages_used} used_pages={int(self.used_pages)}/{int(self.max_pages)}",
                flush=True,
            )
        return pages_used
    
    def _ensure_free_pages(self, ctx: AttentionContext, needed_pages: int, *, avoid_req_id: int | None = None) -> bool:
        needed_pages = int(needed_pages)
        pages_budget = int(getattr(ctx, "pages_per_layer", 0) or 0)
        if pages_budget <= 0:
            pages_budget = int(self.max_pages)
        if int(self.used_pages) + needed_pages <= int(pages_budget):
            return True
        
        candidates = []
        for rid in list(self.running_queue):
            rid = int(rid)
            if avoid_req_id is not None and rid == int(avoid_req_id):
                continue
            req = self.requests.get(rid)
            if req is None or bool(req.get("finished")):
                continue
            pages = int(self._pages_for_running_req(ctx, req))
            candidates.append((pages, rid))
        candidates.sort(reverse=True)
        
        freed = 0
        for pages, rid in candidates:
            if int(self.used_pages) + needed_pages <= int(pages_budget):
                return True
            freed += int(self._pause_req(ctx, rid))
            if freed >= needed_pages:
                break
        
        return bool(int(self.used_pages) + needed_pages <= int(pages_budget))

    def add_request(self, req_id, input_ids, max_tokens, *, want_text: bool = True):
        ids = self._as_np_int64(input_ids)
        full_len = int(len(ids))
        self.requests[req_id] = {
            "prompt_ids": ids,
            "input_ids": ids,
            "prompt_full_len": full_len,
            "prompt_dropped_tokens": 0,
            "generated": [],
            "emitted_len": 0,
            "history_len": 0, 
            "pages_allocated": 0,
            "prefill_cursor": 0,
            "is_prefill": True,
            "finished": False,
            "max_tokens": max_tokens,
            "want_text": bool(want_text),
        }
        self.waiting_queue.append(req_id)

    def step(self, ctx: AttentionContext):
        prof_on = bool(int(os.environ.get("LLMSCHEDULER_PROFILE", "0")))
        if prof_on:
            self._prof_step += 1
            if not hasattr(ctx, "_prof"):
                ctx._prof = {}
            ctx._prof_events = {}
            ctx._prof["batch_ms"] = 0.0
            ctx._prof["trt_ms"] = 0.0
            ctx._prof["sample_ms"] = 0.0
            ctx._prof["meta_ms"] = 0.0
            ctx._prof["kv_write_ms"] = 0.0
            ctx._prof["flashinfer_prefill_ms"] = 0.0
            ctx._prof["flashinfer_decode_ms"] = 0.0
            ctx._prof["kv_write_gpu_ms"] = 0.0
            ctx._prof["flashinfer_gpu_ms"] = 0.0
            ctx._prof["attn_impl_ms"] = 0.0
            ctx._prof["attn_wrap_ms"] = 0.0
            ctx._prof["attn_cast_ms"] = 0.0
            ctx._prof["attn_pe_ms"] = 0.0
            ctx._prof["attn_run_ms"] = 0.0
            ctx._prof["attn_outcast_ms"] = 0.0
            ctx._prof["attn_nan_ms"] = 0.0
            ctx._prof["attn_dtype_in"] = -1.0
            ctx._prof["attn_dtype_out"] = -1.0
            t_step0 = time.perf_counter()

        max_pages = int(getattr(ctx, "pages_per_layer", 0) or 0)
        if max_pages <= 0:
            max_pages = int(self.max_pages)
        page_size = int(self.page_size)
        pages_budget_per_layer = int(max_pages)
        cap_tokens = int(self._cap_tokens_per_req(ctx))
        while self.waiting_queue:
            req_id = self.waiting_queue[0]
            req = self.requests[req_id]
            pages_per_layer = self._ceil_div(int(len(req["input_ids"])), page_size)
            if cap_tokens > 0 and pages_per_layer > int(max_pages):
                new_ids, dropped = self._truncate_to_tokens(self._as_np_int64(req["input_ids"]), cap_tokens)
                req["input_ids"] = new_ids
                if not req.get("generated"):
                    req["prompt_ids"] = new_ids
                    req["prompt_dropped_tokens"] = int(req.get("prompt_dropped_tokens", 0)) + int(dropped)
                else:
                    req["prefix_truncated"] = True
                    req["prefix_dropped_tokens"] = int(req.get("prefix_dropped_tokens", 0)) + int(dropped)
                pages_per_layer = self._ceil_div(int(len(req["input_ids"])), page_size)

            needed = int(pages_per_layer)
            if int(self.used_pages) < max_pages:
                self.waiting_queue.pop(0)
                self.running_queue.append(req_id)
                req["is_prefill"] = True
            else:
                break

        if not self.running_queue:
            return []
        
        if self._log_engine:
            now = time.perf_counter()
            if now - float(self._last_engine_log_t) >= float(self._log_engine_interval_s):
                self._last_engine_log_t = float(now)
                print(
                    f"[engine] step running={len(self.running_queue)} waiting={len(self.waiting_queue)} "
                    f"used_pages={int(self.used_pages)}/{int(self.max_pages)} cap_tokens_per_req={int(cap_tokens)}",
                    flush=True,
                )

        batch_input_ids = []
        ctx_reqs = []
        active_req_ids = []
        active_should_output = []
        pending_updates = []
        batch_tokens = 0
        is_all_decode = True
        active_is_prefill = []
        has_prefill_added = False
        strict_no_mix = bool(int(os.environ.get("LLMSCHEDULER_NO_MIXED_PREFILL_DECODE", "0")))
        prefill_max_chunk = int(os.environ.get("LLMSCHEDULER_PREFILL_MAX_CHUNK", "256"))
        try:
            prefill_budget_frac = float(os.environ.get("LLMSCHEDULER_PREFILL_BUDGET_FRACTION", "0.5"))
        except Exception:
            prefill_budget_frac = 0.5
        prefill_budget_frac = float(max(0.0, min(1.0, prefill_budget_frac)))
        decode_reserve_reqs = int(os.environ.get("LLMSCHEDULER_DECODE_RESERVE_REQS", "32"))
        sched_debug = bool(int(os.environ.get("LLMSCHEDULER_SCHED_STATS", "0")))
        sched_skips = {}

        prefill_items = []
        decode_items = []

        for req_id in list(self.running_queue):
            req = self.requests[req_id]
            if req["is_prefill"]:
                prefill_items.append(int(req_id))
            else:
                decode_items.append(int(req_id))

        ordered_req_ids = list(prefill_items) + list(decode_items)

        use_cpp_sched = bool(int(os.environ.get("LLMSCHEDULER_USE_CPP_SCHEDULER", "1")))
        if not use_cpp_sched:
            raise RuntimeError("Python scheduler is disabled. Set LLMSCHEDULER_USE_CPP_SCHEDULER=1.")
        schedule_step_func = tvm_ffi.get_global_func("custom.schedule_step", allow_missing=True)
        if schedule_step_func is None:
            _ensure_schedule_step_registered()
            schedule_step_func = tvm_ffi.get_global_func("custom.schedule_step")

        if schedule_step_func is not None:
            running_ids = []
            is_prefill_flags = []
            prefill_remaining = []
            decode_ready = []
            for rid in ordered_req_ids:
                r = self.requests[rid]
                running_ids.append(int(rid))
                is_pf = bool(r.get("is_prefill"))
                is_prefill_flags.append(1 if is_pf else 0)
                if is_pf:
                    input_ids = r["input_ids"]
                    prefill_cursor = int(r.get("prefill_cursor", 0))
                    prefill_remaining.append(int(max(0, int(len(input_ids)) - int(prefill_cursor))))
                    decode_ready.append(0)
                else:
                    prefill_remaining.append(0)
                    decode_ready.append(1 if (not bool(r.get("finished")) and bool(r.get("generated"))) else 0)

            running_ids_t = torch.as_tensor(running_ids, dtype=torch.int32, device="cpu")
            is_prefill_t = torch.as_tensor(is_prefill_flags, dtype=torch.int32, device="cpu")
            prefill_rem_t = torch.as_tensor(prefill_remaining, dtype=torch.int32, device="cpu")
            decode_ready_t = torch.as_tensor(decode_ready, dtype=torch.int32, device="cpu")

            ret = schedule_step_func(
                running_ids_t,
                is_prefill_t,
                prefill_rem_t,
                decode_ready_t,
                int(self.max_batch_tokens),
                int(prefill_max_chunk),
                float(prefill_budget_frac),
                int(decode_reserve_reqs),
                1 if bool(strict_no_mix) else 0,
            )

            sel_ids = torch.from_dlpack(ret[0].__dlpack__()).to("cpu", dtype=torch.int32).tolist()
            sel_new_tokens = torch.from_dlpack(ret[1].__dlpack__()).to("cpu", dtype=torch.int32).tolist()
            sel_is_prefill = torch.from_dlpack(ret[2].__dlpack__()).to("cpu", dtype=torch.int32).tolist()
            sel_should_output = None
            if len(ret) > 3:
                sel_should_output = torch.from_dlpack(ret[3].__dlpack__()).to("cpu", dtype=torch.int32).tolist()

            for idx, (req_id, nt, is_pf) in enumerate(zip(sel_ids, sel_new_tokens, sel_is_prefill)):
                req_id = int(req_id)
                req = self.requests[req_id]
                if bool(is_pf):
                    input_ids = req["input_ids"]
                    prefill_cursor = int(req.get("prefill_cursor", 0))
                    if prefill_cursor >= int(len(input_ids)):
                        req["is_prefill"] = False
                        continue
                    remaining = int(len(input_ids) - prefill_cursor)
                    chunk = int(min(int(nt), remaining))
                    if chunk <= 0:
                        sched_skips["prefill_budget"] = int(sched_skips.get("prefill_budget", 0)) + 1
                        continue
                    desired_pages = int(self._ceil_div(int(req.get("history_len", 0)) + int(chunk), page_size))
                    pages_allocated = int(req.get("pages_allocated", 0))
                    add_pages = int(max(0, int(desired_pages) - int(pages_allocated)))
                    if add_pages > 0:
                        if pages_budget_per_layer > 0 and int(desired_pages) > int(pages_budget_per_layer):
                            self._pause_req(ctx, int(req_id))
                            sched_skips["page_budget"] = int(sched_skips.get("page_budget", 0)) + 1
                            continue
                        if int(self.used_pages) + int(add_pages) > max_pages:
                            ok = self._ensure_free_pages(ctx, int(add_pages), avoid_req_id=int(req_id))
                            if not ok:
                                self._pause_req(ctx, int(req_id))
                                sched_skips["page_oom"] = int(sched_skips.get("page_oom", 0)) + 1
                                continue
                        self.used_pages += int(add_pages)
                        req["pages_allocated"] = int(desired_pages)
                    chunk_ids = input_ids[prefill_cursor : prefill_cursor + chunk]
                    total_len = int(req["history_len"]) + int(chunk)
                    batch_input_ids.append(chunk_ids)
                    ctx_reqs.append((total_len, int(chunk)))
                    if sel_should_output is not None:
                        prefill_done = bool(int(sel_should_output[idx]))
                    else:
                        prefill_done = True if strict_no_mix else bool(prefill_cursor + chunk >= int(len(input_ids)))
                    pending_updates.append(("prefill", int(req_id), int(chunk), bool(prefill_done)))
                    active_should_output.append(bool(prefill_done))
                    batch_tokens += int(chunk)
                    is_all_decode = False
                    active_is_prefill.append(True)
                    has_prefill_added = True
                else:
                    if strict_no_mix and has_prefill_added:
                        sched_skips["strict_no_mix"] = int(sched_skips.get("strict_no_mix", 0)) + 1
                        continue
                    if req["finished"]:
                        continue
                    if not req["generated"]:
                        req["is_prefill"] = True
                        continue
                    budget_left = int(self.max_batch_tokens) - int(batch_tokens)
                    if budget_left <= 0:
                        sched_skips["budget"] = int(sched_skips.get("budget", 0)) + 1
                        continue
                    desired_pages = int(self._ceil_div(int(req.get("history_len", 0)) + 1, page_size))
                    pages_allocated = int(req.get("pages_allocated", 0))
                    add_pages = int(max(0, int(desired_pages) - int(pages_allocated)))
                    if add_pages > 0:
                        if pages_budget_per_layer > 0 and int(desired_pages) > int(pages_budget_per_layer):
                            self._pause_req(ctx, int(req_id))
                            sched_skips["page_budget"] = int(sched_skips.get("page_budget", 0)) + 1
                            continue
                        if int(self.used_pages) + int(add_pages) > max_pages:
                            ok = self._ensure_free_pages(ctx, int(add_pages), avoid_req_id=int(req_id))
                            if not ok:
                                self._pause_req(ctx, int(req_id))
                                sched_skips["page_oom"] = int(sched_skips.get("page_oom", 0)) + 1
                                continue
                        self.used_pages += int(add_pages)
                        req["pages_allocated"] = int(desired_pages)
                    last_token = req["generated"][-1]
                    batch_input_ids.append(np.array([last_token], dtype=np.int64))
                    total_len = req["history_len"] + 1
                    new_tokens = 1
                    ctx_reqs.append((total_len, new_tokens))
                    pending_updates.append(("decode", int(req_id), 1, True))
                    active_should_output.append(True)
                    batch_tokens += 1
                    active_is_prefill.append(False)

                active_req_ids.append(req_id)

        if not active_req_ids:
            return []
        if prof_on:
            ctx._prof["batch_ms"] = float((time.perf_counter() - t_step0) * 1000.0)

        flat_input_ids = np.concatenate(batch_input_ids)
        input_tensor = torch.from_numpy(flat_input_ids).to(self.device).long()

        ctx.current_batch_req_ids = active_req_ids
        ctx.current_batch_seq_lens = [nt for _, nt in ctx_reqs]
        ctx.current_batch_total_lens = [tl for tl, _ in ctx_reqs]
        ctx.current_batch_history_lens = [int(tl) - int(nt) for tl, nt in ctx_reqs]
        ctx.current_batch_is_prefill = list(active_is_prefill)
        ctx._engine_step_id = int(getattr(ctx, "_engine_step_id", 0) or 0) + 1
        ctx.is_all_decode = bool(len(active_req_ids) > 0) and all(nt == 1 for _, nt in ctx_reqs)

        if prof_on:
            t_trt0 = time.perf_counter()
        logits = self.runtime.forward(input_tensor, ctx)
        if prof_on:
            try:
                if isinstance(logits, torch.Tensor):
                    torch.cuda.synchronize()
            except Exception:
                pass
            ctx._prof["trt_ms"] = float((time.perf_counter() - t_trt0) * 1000.0)
        
        self.last_step_logits = logits
        if os.environ.get("LLMSCHEDULER_DEBUG_NAN"):
            try:
                if isinstance(logits, torch.Tensor):
                    if bool(torch.isnan(logits).any().item()):
                        print("NaN logits detected", flush=True)
                else:
                    if bool(np.isnan(logits).any()):
                        print("NaN logits detected", flush=True)
            except Exception:
                pass
        
        for kind, rid, delta, done_flag in pending_updates:
            req = self.requests.get(int(rid))
            if req is None or bool(req.get("finished")):
                continue
            if kind == "prefill":
                req["history_len"] = int(req.get("history_len", 0)) + int(delta)
                req["prefill_cursor"] = int(req.get("prefill_cursor", 0)) + int(delta)
                if bool(done_flag) or int(req["prefill_cursor"]) >= int(len(req["input_ids"])):
                    req["is_prefill"] = False
            else:
                req["history_len"] = int(req.get("history_len", 0)) + int(delta)

        start = 0
        step_outputs = []
        sample_logit_indices: list[int] = []
        sample_req_ids: list[int] = []
        for i, req_id in enumerate(active_req_ids):
            req = self.requests[req_id]
            _, new_tokens = ctx_reqs[i]
            logit_idx = start + new_tokens - 1
            start += new_tokens
            if not bool(active_should_output[i]):
                continue
            sample_logit_indices.append(int(logit_idx))
            sample_req_ids.append(int(req_id))

        if not sample_req_ids:
            return []

        if isinstance(logits, torch.Tensor):
            if prof_on:
                t_s0 = time.perf_counter()
            idx = torch.tensor(sample_logit_indices, dtype=torch.int64, device=logits.device)
            token_ids = torch.argmax(logits.index_select(0, idx), dim=-1)
            token_list = [int(x) for x in token_ids.detach().cpu().tolist()]
            if prof_on:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                ctx._prof["sample_ms"] = float((time.perf_counter() - t_s0) * 1000.0)
        else:
            token_list = [int(np.argmax(logits[i])) for i in sample_logit_indices]

        decode_req_pos: list[int] = []
        decode_token_ids: list[int] = []
        want_text_flags: list[bool] = []
        for j, req_id in enumerate(sample_req_ids):
            req_id = int(req_id)
            req = self.requests.get(req_id)
            if req is None or bool(req.get("finished")):
                want_text_flags.append(False)
                continue
            next_token = int(token_list[j])
            req["generated"].append(next_token)
            req["emitted_len"] = int(req.get("emitted_len", 0)) + 1
            want_text = bool(req.get("want_text", True))
            want_text_flags.append(bool(want_text))
            if want_text:
                decode_req_pos.append(len(step_outputs))
                decode_token_ids.append(int(next_token))
            step_outputs.append(
                {
                    "req_id": req_id,
                    "token_id": next_token,
                    "text": "",
                    "finished": False,
                }
            )

            if len(req["generated"]) >= req["max_tokens"]:
                req["finished"] = True
                step_outputs[-1]["finished"] = True
                pages_used = int(req.get("pages_allocated", self._ceil_div(int(req["history_len"]), page_size)))
                self.used_pages = int(max(0, int(self.used_pages) - int(pages_used)))
                req["pages_allocated"] = 0
                if req_id in self.running_queue:
                    try:
                        self.running_queue.remove(req_id)
                    except Exception:
                        pass
                self._free_kv(ctx, int(req_id))
                try:
                    del self.requests[int(req_id)]
                except Exception:
                    pass

        if decode_token_ids:
            f = getattr(self.tokenizer, "batch_decode", None)
            if callable(f):
                decoded = f([[int(t)] for t in decode_token_ids])
            else:
                decoded = [self.tokenizer.decode([int(t)]) for t in decode_token_ids]
            for k, pos in enumerate(decode_req_pos):
                try:
                    step_outputs[int(pos)]["text"] = str(decoded[int(k)])
                except Exception:
                    pass

        if prof_on and (self._prof_step % max(1, int(self._prof_print_interval)) == 0):
            p = getattr(ctx, "_prof", {}) or {}
            ev = getattr(ctx, "_prof_events", {}) or {}
            try:
                for k, pairs in ev.items():
                    if not pairs:
                        continue
                    s = 0.0
                    for a, b in pairs:
                        s += float(a.elapsed_time(b))
                    if k == "kv_write":
                        p["kv_write_gpu_ms"] = float(s)
                    elif k == "flashinfer":
                        p["flashinfer_gpu_ms"] = float(s)
            except Exception:
                pass
            parts = [
                "[profile]",
                f"batch_ms={p.get('batch_ms', 0.0):.2f}",
                f"trt_ms={p.get('trt_ms', 0.0):.2f}",
                f"meta_ms={p.get('meta_ms', 0.0):.2f}",
                f"kv_write_ms={p.get('kv_write_ms', 0.0):.2f}",
                f"kv_write_gpu_ms={p.get('kv_write_gpu_ms', 0.0):.2f}",
                f"flashinfer_prefill_ms={p.get('flashinfer_prefill_ms', 0.0):.2f}",
                f"flashinfer_decode_ms={p.get('flashinfer_decode_ms', 0.0):.2f}",
                f"flashinfer_gpu_ms={p.get('flashinfer_gpu_ms', 0.0):.2f}",
                f"attn_impl_ms={p.get('attn_impl_ms', 0.0):.2f}",
            ]
            for k in ("attn_wrap_ms", "attn_cast_ms", "attn_pe_ms", "attn_run_ms", "attn_outcast_ms", "attn_nan_ms"):
                if k in p:
                    parts.append(f"{k}={float(p.get(k, 0.0)):.2f}")
            for k in ("attn_dtype_in", "attn_dtype_out"):
                if k in p and float(p.get(k, -1.0)) >= 0:
                    parts.append(f"{k}={int(float(p.get(k, 0.0)))}")
            parts.extend(
                [
                    f"sample_ms={p.get('sample_ms', 0.0):.2f}",
                    f"is_all_decode={bool(getattr(ctx, 'is_all_decode', False))}",
                    f"running={len(self.running_queue)}",
                    f"waiting={len(self.waiting_queue)}",
                    f"used_pages={int(self.used_pages)}",
                    f"max_pages={int(self.max_pages)}",
                    f"batch_tokens={int(batch_tokens)}",
                    f"out_reqs={len(sample_req_ids)}",
                ]
            )
            if sched_debug and sched_skips:
                items = sorted(((str(k), int(v)) for k, v in sched_skips.items()), key=lambda x: x[0])
                for k, v in items[:12]:
                    parts.append(f"skip_{k}={int(v)}")
            print(" ".join(parts), flush=True)

        return step_outputs
