from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import onnx
import tensorrt as trt


@dataclass(frozen=True)
class TensorShapeProfile:
    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]


def infer_profiles_from_onnx(
    onnx_path: str,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 8,
    min_seq: int = 1,
    opt_seq: int = 32,
    max_seq: int = 256,
) -> dict[str, TensorShapeProfile]:
    model = onnx.load(onnx_path)
    initializer_names = {i.name for i in model.graph.initializer}

    profiles: dict[str, TensorShapeProfile] = {}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        tt = inp.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims: list[int | None] = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                dims.append(None)

        rank = len(dims)
        if rank == 0:
            continue

        base = [d if d is not None and d > 0 else 1 for d in dims]

        def make(shape_batch: int, shape_seq: int) -> tuple[int, ...]:
            out = list(base)
            if rank >= 1:
                out[0] = shape_batch
            if rank >= 2 and (dims[1] is None or dims[1] <= 0):
                out[1] = shape_seq
            return tuple(int(x) for x in out)

        profiles[inp.name] = TensorShapeProfile(
            min_shape=make(min_batch, min_seq),
            opt_shape=make(opt_batch, opt_seq),
            max_shape=make(max_batch, max_seq),
        )
    return profiles


def get_default_timing_cache_path() -> str:
    import os
    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "orchid", "trt_timing_cache.bin")

def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str | None = None,
    profiles: dict[str, TensorShapeProfile] | None = None,
    profiles_list: list[dict[str, TensorShapeProfile]] | None = None,
    fp16: bool = False,
    workspace_size: int = 2 * 1024 * 1024 * 1024,  # 2GB default
    verbose: bool = False,
) -> bytes:
    import os
    logger = trt.Logger(trt.Logger.WARNING if verbose else trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(onnx_path):
        msgs = []
        for i in range(parser.num_errors):
            err = parser.get_error(i)
            msgs.append(str(err))
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(msgs))

    if verbose:
        print(f"Network has {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")
        for i in range(network.num_outputs):
            print(f"Output {i}: {network.get_output(i).name} {network.get_output(i).shape}")

    try:
        from orchid.llmscheduler.trt.debug import maybe_mark_composite_io_outputs

        maybe_mark_composite_io_outputs(network)
    except Exception:
        pass

    # Build engine
    config = builder.create_builder_config()

    # Timing Cache
    timing_cache_path = os.environ.get("LLMSCHEDULER_TRT_TIMING_CACHE")
    if not timing_cache_path:
        timing_cache_path = get_default_timing_cache_path()
        
    timing_cache = None
    if timing_cache_path:
        if verbose:
            print(f"Using timing cache: {timing_cache_path}")
        try:
            if os.path.exists(timing_cache_path):
                with open(timing_cache_path, "rb") as f:
                    cache_data = f.read()
                    timing_cache = config.create_timing_cache(cache_data)
            else:
                 timing_cache = config.create_timing_cache(b"")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load timing cache: {e}")
            timing_cache = config.create_timing_cache(b"")
    
    if timing_cache:
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # Set memory pool limit for workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_size))

    # Enable FP16 if requested
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        obey = os.environ.get("LLMSCHEDULER_TRT_OBEY_PRECISION_CONSTRAINTS", "").strip().lower() in (
            "1",
            "true",
            "t",
            "yes",
            "y",
            "on",
        )
        try:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS if obey else trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        except AttributeError:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        # Enable TF32 for better FP32 GEMM stability when preferred
        try:
            config.set_flag(trt.BuilderFlag.TF32)
        except Exception:
            pass

        from orchid.llmscheduler.trt.precision_policy import is_composite_attention_layer
        from orchid.llmscheduler.trt.precision_policy import read_policy_from_env
        from orchid.llmscheduler.trt.precision_policy import should_force_fp32

        policy = read_policy_from_env()

        for i in range(network.num_layers):
            layer = network.get_layer(i)

            # Leave explicit casts untouched
            if layer.type == trt.LayerType.CAST:
                continue

            name_l = (layer.name or "").lower()

            force_fp32 = False
            if is_composite_attention_layer(layer.type, name_l):
                try:
                    layer.precision = trt.DataType.HALF
                    try:
                        for j in range(getattr(layer, "num_inputs", 0)):
                            if hasattr(layer, "set_input_type"):
                                layer.set_input_type(j, trt.DataType.HALF)
                    except Exception:
                        pass
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.DataType.HALF)
                    if verbose:
                        print(f"  Layer {i} ({layer.name}, {layer.type}): set plugin I/O to FP16")
                except Exception as e:
                    if verbose:
                        print(f"  Layer {i} ({layer.name}, {layer.type}): set plugin FP16 failed: {e}")
                # Do not apply FP32 forcing for the plugin; move to next layer
                continue
            force_fp32 = should_force_fp32(layer.type, name_l, policy)

            if force_fp32:
                try:
                    layer.precision = trt.DataType.FLOAT
                    # Try to force input types if API is available (TRT10+)
                    try:
                        for j in range(getattr(layer, "num_inputs", 0)):
                            if hasattr(layer, "set_input_type"):
                                layer.set_input_type(j, trt.DataType.FLOAT)
                    except Exception:
                        pass
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.DataType.FLOAT)
                    if verbose:
                        print(f"  Layer {i} ({layer.name}, {layer.type}): forced to FP32")
                except Exception as e:
                    if verbose:
                        print(f"  Layer {i} ({layer.name}, {layer.type}): FP32 force failed: {e}")

    if profiles_list is None:
        if profiles is None:
            profiles = infer_profiles_from_onnx(onnx_path)
        profiles_list = [profiles]

    for idx, prof in enumerate(profiles_list):
        profile = builder.create_optimization_profile()
        for name, p in prof.items():
            res = profile.set_shape(name, p.min_shape, p.opt_shape, p.max_shape)
            if res is False:
                raise RuntimeError(f"Failed to set optimization profile shape for {name} (profile={idx})")
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)

    # Save Timing Cache
    if timing_cache and timing_cache_path:
        try:
            os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
            with open(timing_cache_path, "wb") as f:
                f.write(timing_cache.serialize())
            if verbose:
                print(f"Updated timing cache: {timing_cache_path}")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to save timing cache to {timing_cache_path}: {e}")

    if serialized is None:
        raise RuntimeError("Failed to build TensorRT engine")
    engine_bytes = bytes(serialized)

    if engine_path is not None:
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

    return engine_bytes

class EngineBuilder:
    def build(self, onnx_path, fp16=True, input_profile=None, verbose: bool = False, engine_path=None):
        import os
        import time

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

        log_progress = _env_bool("LLMSCHEDULER_ENGINE_PROGRESS", True)

        def should_rebuild(onnx_p, engine_p):
            if not engine_p or not os.path.exists(engine_p):
                return True
            try:
                # If ONNX is newer than Engine, rebuild
                if os.path.getmtime(onnx_p) > os.path.getmtime(engine_p):
                    if log_progress:
                        print(f"[trt] ONNX model is newer than engine, rebuilding...")
                    return True
            except Exception:
                return True
            return False

        if not should_rebuild(onnx_path, engine_path):
            if log_progress:
                try:
                    sz = os.path.getsize(engine_path)
                except Exception:
                    sz = -1
                print(f"[trt] load engine from {engine_path} size={sz}")
            with open(engine_path, "rb") as f:
                return f.read()

        if log_progress:
            print(f"[trt] build engine from {onnx_path} -> {engine_path or '(memory)'} fp16={bool(fp16)}")
        t0 = time.perf_counter()

        profiles_list = None
        profiles = None
        if input_profile:
            profiles_list = []
            for p in input_profile:
                prof: dict[str, TensorShapeProfile] = {}
                for name, shapes in p.items():
                    prof[name] = TensorShapeProfile(
                        min_shape=tuple(shapes[0]),
                        opt_shape=tuple(shapes[1]),
                        max_shape=tuple(shapes[2]),
                    )
                profiles_list.append(prof)

        out = build_engine_from_onnx(
            onnx_path,
            engine_path=engine_path,
            profiles=profiles,
            profiles_list=profiles_list,
            fp16=fp16,
            verbose=verbose,
        )
        if log_progress:
            dt = time.perf_counter() - t0
            try:
                sz = os.path.getsize(engine_path) if engine_path else len(out)
            except Exception:
                sz = -1
            print(f"[trt] engine ready dt_s={dt:.2f} size={sz}")
        return out
