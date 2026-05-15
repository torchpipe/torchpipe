from __future__ import annotations

import os
import pathlib
from typing import Any

import numpy as np


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


def sanitize_tensor_name(name: str) -> str:
    return (
        name.replace(os.sep, "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def maybe_mark_composite_io_outputs(network: Any) -> None:
    if not _env_bool("LLMSCHEDULER_DEBUG_MARK_COMPOSITE_IO", False):
        return
    try:
        max_mark = int(os.environ.get("LLMSCHEDULER_DEBUG_MARK_COMPOSITE_MAX", "1"))
    except Exception:
        max_mark = 1

    try:
        import tensorrt as trt

        plugin_types = {trt.LayerType.PLUGIN_V2, trt.LayerType.PLUGIN_V3}
    except Exception:
        plugin_types = set()

    marked = 0
    for i in range(int(getattr(network, "num_layers", 0))):
        layer = network.get_layer(i)
        name_l = (getattr(layer, "name", "") or "").lower()
        layer_type = getattr(layer, "type", None)
        is_plugin = layer_type in plugin_types if plugin_types else ("plugin" in str(layer_type).lower())
        if not (is_plugin and "composite_attention" in name_l):
            continue

        tensors = []
        for j in (0, 1, 2):
            try:
                tensors.append(layer.get_input(j))
            except Exception:
                pass
        try:
            tensors.append(layer.get_output(0))
        except Exception:
            pass

        for t in tensors:
            if t is None:
                continue
            try:
                network.mark_output(t)
            except Exception:
                pass

        marked += 1
        if marked >= max_mark:
            break


def dump_named_arrays(outputs: dict[str, Any], dump_dir: str, step: int) -> list[str]:
    out_dir = pathlib.Path(dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name, arr in outputs.items():
        safe = sanitize_tensor_name(str(name))
        path = out_dir / f"step{step}_{safe}.npy"
        np.save(path, arr)
        written.append(str(path))
    return written

