from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


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


@dataclass(frozen=True)
class PrecisionPolicy:
    fp32_norm: bool
    fp32_softmax: bool
    fp32_norm_subgraph: bool
    fp32_logits_matmul: bool
    fp32_proj_matmul: bool
    norm_patterns: tuple[str, ...]
    logits_patterns: tuple[str, ...]
    proj_patterns: tuple[str, ...]


def read_policy_from_env() -> PrecisionPolicy:
    fp32_norm = _env_bool("LLMSCHEDULER_FP32_NORM", True)
    fp32_softmax = _env_bool("LLMSCHEDULER_FP32_SOFTMAX", True)
    fp32_norm_subgraph = _env_bool("LLMSCHEDULER_FP32_NORM_SUBGRAPH", True)
    fp32_logits_matmul = _env_bool("LLMSCHEDULER_FP32_LOGITS_MATMUL", False)
    fp32_proj_matmul = _env_bool("LLMSCHEDULER_FP32_PROJ_MATMUL", False)

    norm_patterns = ["norm", "layernorm", "ln", "rms"]
    logits_patterns = ["lm_head", "logits", "classifier"]
    proj_patterns = ["q_proj", "k_proj", "v_proj", "o_proj"]

    extra = os.environ.get("LLMSCHEDULER_FP32_LAYER_NAME_PATTERNS", "")
    extra_patterns = [p.strip().lower() for p in extra.split(",") if p.strip()]
    norm_patterns = list(dict.fromkeys([p.lower() for p in norm_patterns + extra_patterns]))
    logits_patterns = list(dict.fromkeys([p.lower() for p in logits_patterns + extra_patterns]))

    return PrecisionPolicy(
        fp32_norm=fp32_norm,
        fp32_softmax=fp32_softmax,
        fp32_norm_subgraph=fp32_norm_subgraph,
        fp32_logits_matmul=fp32_logits_matmul,
        fp32_proj_matmul=fp32_proj_matmul,
        norm_patterns=tuple(norm_patterns),
        logits_patterns=tuple(logits_patterns),
        proj_patterns=tuple(proj_patterns),
    )


def _layer_type_name(layer_type: Any) -> str:
    try:
        return str(layer_type).split(".")[-1].upper()
    except Exception:
        return str(layer_type).upper()


def is_composite_attention_layer(layer_type: Any, name_l: str) -> bool:
    if "composite_attention" not in name_l:
        return False
    try:
        import tensorrt as trt

        return layer_type in (trt.LayerType.PLUGIN_V2, trt.LayerType.PLUGIN_V3)
    except Exception:
        return "PLUGIN" in _layer_type_name(layer_type)


def should_force_fp32(layer_type: Any, name_l: str, policy: PrecisionPolicy) -> bool:
    lt = _layer_type_name(layer_type)
    if lt == "SOFTMAX":
        return policy.fp32_softmax
    if lt == "NORMALIZATION":
        return policy.fp32_norm
    if lt in ("REDUCE", "ELEMENTWISE", "SCALE"):
        return policy.fp32_norm_subgraph and any(p in name_l for p in policy.norm_patterns)
    if lt in ("MATRIX_MULTIPLY", "MATMUL"):
        return (policy.fp32_logits_matmul and any(p in name_l for p in policy.logits_patterns)) or (
            policy.fp32_proj_matmul and any(p in name_l for p in policy.proj_patterns)
        )
    return False

