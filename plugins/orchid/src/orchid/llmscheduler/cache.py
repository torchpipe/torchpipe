from __future__ import annotations

import os
import re


def user_cache_dir(app_name: str = "orchid") -> str:
    base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, app_name)


def _slug(s: str) -> str:
    s = s.strip()
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "model"


def cached_model_dir(model_id: str, *, dtype: str = "fp16", app_name: str = "orchid") -> str:
    return os.path.join(user_cache_dir(app_name), "models", _slug(model_id), _slug(dtype))


def cached_onnx_path(model_id: str, *, dtype: str = "fp16", app_name: str = "orchid") -> str:
    return os.path.join(cached_model_dir(model_id, dtype=dtype, app_name=app_name), "model.onnx")


def cached_composite_onnx_path(model_id: str, *, dtype: str = "fp16", app_name: str = "orchid") -> str:
    return os.path.join(cached_model_dir(model_id, dtype=dtype, app_name=app_name), "model.composite.onnx")


def cached_trt_engine_path(model_id: str, *, dtype: str = "fp16", app_name: str = "orchid") -> str:
    return os.path.join(cached_model_dir(model_id, dtype=dtype, app_name=app_name), "model.plan")

