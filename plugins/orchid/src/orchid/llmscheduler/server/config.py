from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    test_mode: bool
    model_path: str
    tokenizer_path: str
    use_fp16: bool
    engine_path: str | None
    num_layers: int | None
    num_heads: int | None
    kv_num_heads: int | None
    head_dim: int | None
    page_size: int
    max_pages: int | None


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def load_config_from_env() -> ServerConfig:
    host = os.environ.get("LLMSCHEDULER_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("LLMSCHEDULER_PORT", "8000"))
    except Exception:
        port = 8000

    from ..cache import cached_composite_onnx_path
    from ..cache import cached_trt_engine_path

    test_mode = _env_bool("LLMSCHEDULER_TEST_MODE", False)
    model_path = os.environ.get("MODEL_PATH", cached_composite_onnx_path("Qwen/Qwen3-0.6B", dtype="fp16"))
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen3-0.6B")
    use_fp16 = _env_bool("LLMSCHEDULER_USE_FP16", True)
    engine_path = os.environ.get("ENGINE_PATH") or os.environ.get("LLMSCHEDULER_ENGINE_PATH") or cached_trt_engine_path("Qwen/Qwen3-0.6B", dtype="fp16")
    num_layers = os.environ.get("LLMSCHEDULER_NUM_LAYERS")
    num_heads = os.environ.get("LLMSCHEDULER_NUM_HEADS")
    kv_num_heads = os.environ.get("LLMSCHEDULER_KV_NUM_HEADS")
    head_dim = os.environ.get("LLMSCHEDULER_HEAD_DIM")
    page_size = os.environ.get("LLMSCHEDULER_PAGE_SIZE", "16")
    max_pages = os.environ.get("LLMSCHEDULER_MAX_PAGES")

    return ServerConfig(
        host=host,
        port=int(port),
        test_mode=bool(test_mode),
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        use_fp16=bool(use_fp16),
        engine_path=str(engine_path) if engine_path else None,
        num_layers=int(num_layers) if num_layers else None,
        num_heads=int(num_heads) if num_heads else None,
        kv_num_heads=int(kv_num_heads) if kv_num_heads else None,
        head_dim=int(head_dim) if head_dim else None,
        page_size=int(page_size),
        max_pages=int(max_pages) if max_pages else None,
    )
