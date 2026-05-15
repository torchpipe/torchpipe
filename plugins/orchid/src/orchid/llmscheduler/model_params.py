from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelParams:
    num_layers: int
    num_heads: int
    kv_num_heads: int
    head_dim: int
    page_size: int
    max_pages: int


def _infer_from_composite_onnx(model_path: str) -> dict[str, int]:
    import onnx

    m = onnx.load(model_path)
    nodes = [n for n in m.graph.node if n.domain == "com.custom.llm" and n.op_type == "CompositeAttention"]
    if not nodes:
        return {}
    a0 = {a.name: a for a in nodes[0].attribute}
    return {
        "num_layers": int(len(nodes)),
        "num_heads": int(a0["q_num_heads"].i),
        "kv_num_heads": int(a0["kv_num_heads"].i),
        "head_dim": int(a0["head_dim"].i),
    }


def _infer_from_hf_config(tokenizer_path: str) -> dict[str, int]:
    cfg: dict[str, Any] | None = None
    if os.path.isdir(tokenizer_path):
        p = os.path.join(tokenizer_path, "config.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                cfg = json.load(f)
    if cfg is None:
        try:
            from transformers import AutoConfig

            cfg_obj = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
            cfg = cfg_obj.to_dict()
        except Exception:
            return {}

    num_layers = cfg.get("num_hidden_layers")
    num_heads = cfg.get("num_attention_heads")
    hidden_size = cfg.get("hidden_size")
    kv_num_heads = cfg.get("num_key_value_heads", num_heads)
    if not (num_layers and num_heads and hidden_size):
        return {}
    head_dim = int(cfg.get("head_dim", int(hidden_size) // int(num_heads)))
    return {
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "kv_num_heads": int(kv_num_heads),
        "head_dim": int(head_dim),
    }


def infer_model_params(
    model_path: str | None,
    tokenizer_path: str | None,
    *,
    env: dict[str, str] | None = None,
    default_page_size: int = 16,
    default_max_pages: int = 4096,
) -> ModelParams:
    e = env if env is not None else os.environ

    inferred: dict[str, int] = {}
    if model_path:
        try:
            inferred.update(_infer_from_composite_onnx(model_path))
        except Exception:
            pass
    if tokenizer_path:
        try:
            for k, v in _infer_from_hf_config(tokenizer_path).items():
                inferred.setdefault(k, v)
        except Exception:
            pass

    def env_int(name: str) -> int | None:
        v = e.get(name)
        if v is None:
            return None
        try:
            return int(str(v))
        except Exception:
            return None

    num_layers = env_int("LLMSCHEDULER_NUM_LAYERS") or inferred.get("num_layers") or 28
    num_heads = env_int("LLMSCHEDULER_NUM_HEADS") or inferred.get("num_heads") or 16
    kv_num_heads = env_int("LLMSCHEDULER_KV_NUM_HEADS") or inferred.get("kv_num_heads") or 8
    head_dim = env_int("LLMSCHEDULER_HEAD_DIM") or inferred.get("head_dim") or 128
    page_size = env_int("LLMSCHEDULER_PAGE_SIZE") or int(default_page_size)
    max_pages = env_int("LLMSCHEDULER_MAX_PAGES") or int(default_max_pages)

    return ModelParams(
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        kv_num_heads=int(kv_num_heads),
        head_dim=int(head_dim),
        page_size=int(page_size),
        max_pages=int(max_pages),
    )

