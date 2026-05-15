from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.responses import StreamingResponse

from .config import ServerConfig
from .protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseUsage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
)

def _model_json(obj: Any) -> str:
    f = getattr(obj, "model_dump_json", None)
    if callable(f):
        return f()
    return obj.json()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _test_mode_response(request: ChatCompletionRequest):
    if request.stream:
        async def event_generator():
            chunk = ChatCompletionStreamResponse(
                model=request.model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta={"content": "ok"},
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {_model_json(chunk)}\n\n"
            final_chunk = ChatCompletionStreamResponse(
                model=request.model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta={},
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {_model_json(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    content = "ok"
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionResponseUsage(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        ),
    )


def _message_text(m: ChatCompletionMessage) -> str:
    c = m.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for item in c:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
        return "".join(parts)
    return ""


def _decode_token_chunk(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    f = getattr(tokenizer, "batch_decode", None)
    if callable(f):
        try:
            decoded = f(
                [list(int(t) for t in token_ids)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded:
                return str(decoded[0])
        except TypeError:
            try:
                decoded = f([list(int(t) for t in token_ids)])
                if decoded:
                    return str(decoded[0])
            except Exception:
                pass
        except Exception:
            pass
    g = getattr(tokenizer, "decode", None)
    if callable(g):
        try:
            return str(g(list(int(t) for t in token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False))
        except TypeError:
            try:
                return str(g(list(int(t) for t in token_ids)))
            except Exception:
                pass
        except Exception:
            pass
    return ""


def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI()

    state: dict[str, Any] = {"engine": None, "tokenizer": None, "ctx": None}

    @app.get("/health")
    async def health():
        mode = "test" if config.test_mode else "trt"
        ready = True if config.test_mode else (state.get("engine") is not None)
        return {"status": "ok", "mode": mode, "ready": bool(ready)}
    
    @app.get("/metrics")
    async def metrics():
        ctx = state.get("ctx")
        if ctx is None:
            return Response(content="", media_type="text/plain; charset=utf-8")
        try:
            max_pages = int(getattr(ctx, "max_pages", 0))
        except Exception:
            max_pages = 0
        try:
            page_size = int(getattr(ctx, "page_size", 0))
        except Exception:
            page_size = 0
        try:
            num_layers = int(getattr(ctx, "num_layers", 0))
        except Exception:
            num_layers = 0
        body = (
            "# TYPE llmscheduler_ready gauge\n"
            f"llmscheduler_ready {1 if state.get('engine') is not None else 0}\n"
            "# TYPE llmscheduler_max_pages gauge\n"
            f"llmscheduler_max_pages {max_pages}\n"
            "# TYPE llmscheduler_page_size gauge\n"
            f"llmscheduler_page_size {page_size}\n"
            "# TYPE llmscheduler_num_layers gauge\n"
            f"llmscheduler_num_layers {num_layers}\n"
        )
        return Response(content=body, media_type="text/plain; charset=utf-8")

    @app.on_event("startup")
    async def startup_event():
        if config.test_mode:
            return

        from transformers import AutoTokenizer
        from ..model_params import infer_model_params
        from ..runtime.trt_runtime import TensorRTModelRuntime
        from ..core.engine import ContinuousBatchingEngine
        from ..runtime.base import AttentionContext
        from .serving_engine import ServingEngine

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

        p = infer_model_params(config.model_path, config.tokenizer_path)
        num_layers = int(config.num_layers) if config.num_layers is not None else int(p.num_layers)
        num_heads = int(config.num_heads) if config.num_heads is not None else int(p.num_heads)
        kv_num_heads = int(config.kv_num_heads) if config.kv_num_heads is not None else int(p.kv_num_heads)
        head_dim = int(config.head_dim) if config.head_dim is not None else int(p.head_dim)
        page_size = int(config.page_size) if config.page_size else int(p.page_size)

        def load_model():
            return TensorRTModelRuntime(config.model_path, use_fp16=config.use_fp16, engine_path=config.engine_path)

        trt_runtime = await asyncio.to_thread(load_model)
        
        max_pages = config.max_pages
        if max_pages is None or int(max_pages) <= 0:
            try:
                import torch
                free_b, total_b = torch.cuda.mem_get_info()
                util = float(os.environ.get("LLMSCHEDULER_GPU_MEMORY_UTILIZATION", "0.85"))
                reserve_mb = int(os.environ.get("LLMSCHEDULER_KV_CACHE_RESERVED_MB", "4096"))
                reserve_b = int(reserve_mb) * 1024 * 1024
                per_page_b = int(4) * int(page_size) * int(kv_num_heads) * int(head_dim)
                budget_b = int(float(free_b) * float(util)) - int(reserve_b)
                if per_page_b <= 0 or budget_b <= 0:
                    raise RuntimeError(f"invalid kv budget: per_page_b={per_page_b} budget_b={budget_b}")
                auto_pages = int(budget_b // per_page_b)
                step = int(os.environ.get("LLMSCHEDULER_MAX_PAGES_ROUND", "256"))
                if step > 0:
                    auto_pages = int(auto_pages // step) * step
                min_pages = int(os.environ.get("LLMSCHEDULER_MAX_PAGES_MIN", "1024"))
                auto_pages = int(max(auto_pages, min_pages))
                max_pages_cap = int(os.environ.get("LLMSCHEDULER_MAX_PAGES_MAX", "262144"))
                auto_pages = int(min(auto_pages, max_pages_cap))
                max_pages = int(auto_pages)
                if os.environ.get("LLMSCHEDULER_LOG_STARTUP"):
                    kv_gb = (float(max_pages) * float(per_page_b)) / (1024**3)
                    print(
                        f"[server] auto max_pages={max_pages} per_page_bytes={per_page_b} kv_cache_gb={kv_gb:.2f} "
                        f"free_gb={float(free_b)/(1024**3):.2f} util={util} reserve_mb={reserve_mb}",
                        flush=True,
                    )
            except Exception as e:
                max_pages = int(p.max_pages)
                if os.environ.get("LLMSCHEDULER_LOG_STARTUP"):
                    print(f"[server] auto max_pages failed, fallback={max_pages} err={e}", flush=True)
        max_pages = int(max_pages)

        ctx = AttentionContext(
            num_layers,
            num_heads,
            kv_num_heads,
            head_dim,
            page_size,
            max_pages,
            use_cpp_metadata=True,
            device="cuda",
            use_fp16=config.use_fp16,
        )
        await asyncio.to_thread(trt_runtime.init_runtime, ctx)

        core_engine = ContinuousBatchingEngine(trt_runtime, tokenizer, max_pages=max_pages, page_size=page_size)
        engine = ServingEngine(core_engine, ctx)
        engine.start()

        state["engine"] = engine
        state["tokenizer"] = tokenizer
        state["ctx"] = ctx
        if os.environ.get("LLMSCHEDULER_LOG_STARTUP"):
            print(
                f"[server] startup mode=trt model_path={config.model_path} engine_path={config.engine_path} "
                f"page_size={page_size} max_pages={max_pages} use_fp16={bool(config.use_fp16)}",
                flush=True,
            )

    @app.on_event("shutdown")
    async def shutdown_event():
        eng = state.get("engine")
        if eng is not None:
            eng.stop()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if config.test_mode:
            return _test_mode_response(request)

        log_req = os.environ.get("LLMSCHEDULER_LOG_REQUESTS")
        t0 = time.perf_counter()

        engine = state.get("engine")
        tokenizer = state.get("tokenizer")
        if engine is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="Engine not initialized")

        import numpy as np
        if request.input_ids is not None:
            input_ids = np.array(list(request.input_ids), dtype=np.int64)
        else:
            apply_template = bool(int(os.environ.get("LLMSCHEDULER_APPLY_CHAT_TEMPLATE", "0")))
            if apply_template:
                msgs = [{"role": str(m.role), "content": _message_text(m)} for m in request.messages]
                f = getattr(tokenizer, "apply_chat_template", None)
                if callable(f):
                    try:
                        ids = f(msgs, add_generation_prompt=True, tokenize=True)
                        input_ids = np.array(list(ids), dtype=np.int64)
                    except Exception:
                        prompt = "\n".join([_message_text(m) for m in request.messages])
                        input_ids = np.array(tokenizer.encode(prompt), dtype=np.int64)
                else:
                    prompt = "\n".join([_message_text(m) for m in request.messages])
                    input_ids = np.array(tokenizer.encode(prompt), dtype=np.int64)
            else:
                prompt = "\n".join([_message_text(m) for m in request.messages])
                input_ids = np.array(tokenizer.encode(prompt), dtype=np.int64)
        max_tokens = request.max_tokens or request.max_completion_tokens or 128
        trace_id = str(uuid.uuid4())
        want_text = (not bool(request.stream)) and (not bool(request.skip_detokenize))
        queue = await engine.submit(input_ids, max_tokens, want_text=want_text)
        if log_req:
            print(
                f"[server] trace_id={trace_id} prompt_tokens={len(input_ids)} max_tokens={int(max_tokens)} stream={bool(request.stream)}",
                flush=True,
            )

        if request.stream:
            try:
                stream_flush_tokens = int(os.environ.get("LLMSCHEDULER_STREAM_FLUSH_TOKENS", "1"))
            except Exception:
                stream_flush_tokens = 1
            stream_flush_tokens = max(1, int(stream_flush_tokens))

            async def event_generator():
                had_error = False
                completion_tokens = 0
                buffered_token_ids: list[int] = []

                def flush_buffer() -> str:
                    nonlocal buffered_token_ids
                    text = _decode_token_chunk(tokenizer, buffered_token_ids)
                    buffered_token_ids = []
                    return text

                while True:
                    output = await queue.get()
                    if output is None:
                        break
                    if isinstance(output, dict) and output.get("error"):
                        token_text = flush_buffer()
                        if token_text:
                            yield "data: " + _json_dumps(
                                {
                                    "model": request.model,
                                    "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
                                }
                            ) + "\n\n"
                        had_error = True
                        if log_req:
                            dt = time.perf_counter() - t0
                            print(
                                f"[server] trace_id={trace_id} finish=error completion_tokens={completion_tokens} dt_s={dt:.3f} detail={output.get('error')}",
                                flush=True,
                            )
                        chunk = ChatCompletionStreamResponse(
                            model=request.model,
                            choices=[
                                ChatCompletionStreamResponseChoice(
                                    index=0,
                                    delta={"content": str(output.get("error"))},
                                    finish_reason="error",
                                )
                            ],
                        )
                        yield f"data: {_model_json(chunk)}\n\n"
                        break
                    completion_tokens += 1
                    buffered_token_ids.append(int(output.get("token_id", 0)))
                    if len(buffered_token_ids) < stream_flush_tokens and not bool(output.get("finished")):
                        continue
                    token_text = flush_buffer()
                    if token_text:
                        yield "data: " + _json_dumps(
                            {
                                "model": request.model,
                                "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
                            }
                        ) + "\n\n"

                token_text = flush_buffer()
                if token_text:
                    yield "data: " + _json_dumps(
                        {
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
                        }
                    ) + "\n\n"

                final_chunk = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta={},
                            finish_reason="error" if had_error else "stop",
                        )
                    ],
                )
                yield f"data: {_model_json(final_chunk)}\n\n"
                if not had_error:
                    usage_chunk = ChatCompletionStreamResponse(
                        model=request.model,
                        usage=ChatCompletionResponseUsage(
                            prompt_tokens=len(input_ids),
                            completion_tokens=int(completion_tokens),
                            total_tokens=int(len(input_ids) + completion_tokens),
                        ),
                    )
                    yield f"data: {_model_json(usage_chunk)}\n\n"
                    if log_req:
                        dt = time.perf_counter() - t0
                        print(
                            f"[server] trace_id={trace_id} finish=stop completion_tokens={completion_tokens} dt_s={dt:.3f}",
                            flush=True,
                        )
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        generated_text = ""
        generated_token_ids = []
        completion_tokens = 0
        while True:
            output = await queue.get()
            if output is None:
                break
            if isinstance(output, dict) and output.get("error"):
                if log_req:
                    dt = time.perf_counter() - t0
                    print(
                        f"[server] trace_id={trace_id} finish=error completion_tokens={completion_tokens} dt_s={dt:.3f} detail={output.get('error')}",
                        flush=True,
                    )
                raise HTTPException(status_code=503, detail=str(output.get("error")))
            if bool(request.skip_detokenize):
                generated_token_ids.append(int(output.get("token_id", 0)))
            else:
                generated_text += str(output.get("text", ""))
            completion_tokens += 1
        if bool(request.skip_detokenize):
            try:
                generated_text = str(tokenizer.decode(list(generated_token_ids)))
            except Exception:
                generated_text = ""
        if log_req:
            dt = time.perf_counter() - t0
            print(
                f"[server] trace_id={trace_id} finish=stop completion_tokens={completion_tokens} dt_s={dt:.3f}",
                flush=True,
            )

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=generated_text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionResponseUsage(
                prompt_tokens=len(input_ids),
                completion_tokens=int(completion_tokens),
                total_tokens=int(len(input_ids) + completion_tokens),
            ),
        )

    return app
