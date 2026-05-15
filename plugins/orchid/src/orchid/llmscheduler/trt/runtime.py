from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorrt as trt

# Import plugin to register it with TensorRT
import orchid.llmscheduler.plugins.trt_plugins
import os
import json
from collections import OrderedDict, defaultdict

from ._cuda import DeviceBuffer, create_stream, destroy_stream, malloc, memcpy_dtoh_async, memcpy_htod_async, set_device, stream_synchronize
from .builder import TensorShapeProfile, build_engine_from_onnx, infer_profiles_from_onnx


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: np.dtype
    is_input: bool


def _as_np_dtype(dt: trt.DataType) -> np.dtype:
    return np.dtype(trt.nptype(dt))


def _torch_ptr_and_dtype(x):
    try:
        import torch
    except Exception:
        return None
    if not isinstance(x, torch.Tensor):
        return None
    if not x.is_cuda:
        raise ValueError("torch.Tensor input must be on CUDA")
    if not x.is_contiguous():
        x = x.contiguous()
    return int(x.data_ptr()), np.dtype(str(x.detach().cpu().numpy().dtype))


class TensorRTRuntime:
    def __init__(
        self,
        engine_path: str | None = None,
        engine_bytes: bytes | None = None,
        onnx_path: str | None = None,
        profiles: dict[str, TensorShapeProfile] | None = None,
        device_id: int = 0,
        fp16: bool = False,
        ctx: Any = None
    ):
        self._ctx = ctx
        if ctx:
             from orchid.llmscheduler.plugins.composite_attention import set_context
             set_context(ctx)
        if engine_path is None and engine_bytes is None:
            if onnx_path is None:
                raise ValueError("Provide engine_path, engine_bytes, or onnx_path")
            if profiles is None:
                profiles = infer_profiles_from_onnx(onnx_path)
            engine_bytes = build_engine_from_onnx(onnx_path, None, profiles=profiles, fp16=fp16)

        if engine_bytes is None and engine_path is not None:
            with open(engine_path, "rb") as f:
                engine_bytes = f.read()

        if engine_bytes is None:
            raise ValueError("Failed to load engine bytes")

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self._layer_profile_on = bool(int(os.environ.get("LLMSCHEDULER_TRT_LAYER_PROFILE", "0")))
        self._layer_profile_topk = int(os.environ.get("LLMSCHEDULER_TRT_LAYER_PROFILE_TOPK", "20"))
        self._layer_profile_interval = int(os.environ.get("LLMSCHEDULER_TRT_LAYER_PROFILE_INTERVAL", "200"))
        self._layer_profile_dump = os.environ.get("LLMSCHEDULER_TRT_LAYER_PROFILE_DUMP", "").strip()
        self._layer_profile_step = 0
        self._layer_profiler = None
        if self._layer_profile_on:
            class _LayerProfiler(trt.IProfiler):
                def __init__(self):
                    super().__init__()
                    self.times = {}
                    self.calls = 0

                def reset(self):
                    self.times = {}
                    self.calls = 0

                def report_layer_time(self, layer_name: str, ms: float):
                    self.times[str(layer_name)] = float(ms)
                    self.calls += 1

            self._layer_profiler = _LayerProfiler()
            try:
                self._context.profiler = self._layer_profiler
            except Exception:
                pass

        set_device(device_id)
        self._stream = create_stream()
        self._buffers: dict[tuple[str, int], DeviceBuffer] = {}
        self._torch_output_buffers: dict[str, Any] = {}
        self._shape_trace_on = bool(int(os.environ.get("LLMSCHEDULER_TRT_SHAPE_TRACE", "0")))
        self._shape_trace_interval = int(os.environ.get("LLMSCHEDULER_TRT_SHAPE_TRACE_INTERVAL", "200"))
        self._shape_trace_topk = int(os.environ.get("LLMSCHEDULER_TRT_SHAPE_TRACE_TOPK", "8"))
        self._shape_trace_dump = os.environ.get("LLMSCHEDULER_TRT_SHAPE_DUMP", "")
        self._shape_trace_step = 0
        self._shape_trace_stats: dict[str, dict[str, Any]] = {}
        self._active_profile = -1
        self._num_profiles = int(getattr(self._engine, "num_optimization_profiles", 1) or 1)

        self._tensors: dict[str, TensorInfo] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            dt = self._engine.get_tensor_dtype(name)
            self._tensors[name] = TensorInfo(
                name=name,
                dtype=_as_np_dtype(dt),
                is_input=(mode == trt.TensorIOMode.INPUT),
            )

        env_spec = os.environ.get("LLMSCHEDULER_TRT_INPUT_IDS_PROFILES", "")
        parsed_specs = self._parse_input_ids_profiles(env_spec)
        engine_specs: list[tuple[int, int, int]] | None = None
        try:
            if "input_ids" in self._tensors and hasattr(self._engine, "get_tensor_profile_shape"):
                specs: list[tuple[int, int, int]] = []
                for i in range(int(self._num_profiles)):
                    mn, opt, mx = self._engine.get_tensor_profile_shape("input_ids", int(i))
                    mn0 = int(mn[0]) if mn and int(len(mn)) >= 1 else 0
                    opt0 = int(opt[0]) if opt and int(len(opt)) >= 1 else 0
                    mx0 = int(mx[0]) if mx and int(len(mx)) >= 1 else 0
                    if mn0 > 0 and opt0 > 0 and mx0 > 0 and mn0 <= opt0 <= mx0:
                        specs.append((mn0, opt0, mx0))
                if len(specs) == int(self._num_profiles):
                    engine_specs = specs
        except Exception:
            engine_specs = None

        if engine_specs is not None:
            ok = False
            if parsed_specs and len(parsed_specs) == int(self._num_profiles):
                ok = True
                for i in range(int(self._num_profiles)):
                    a = parsed_specs[i]
                    b = engine_specs[i]
                    if int(a[0]) != int(b[0]) or int(a[2]) != int(b[2]):
                        ok = False
                        break
            self._profile_specs = parsed_specs if ok else list(engine_specs)
        else:
            self._profile_specs = parsed_specs

        if self._profile_specs and len(self._profile_specs) != self._num_profiles:
            self._profile_specs = self._profile_specs[: self._num_profiles]

    @staticmethod
    def _parse_input_ids_profiles(spec: str) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        s = (spec or "").strip()
        if not s:
            # Default profiles if not specified
            # 5 disjoint profiles covering typical ranges
            s = "1,32,128;129,256,512;513,1024,2048;2049,3072,4096;4097,6144,8192"
        
        for part in s.split(";"):
            p = part.strip()
            if not p:
                continue
            cols = [c.strip() for c in p.split(",")]
            if len(cols) != 3:
                continue
            try:
                mn, opt, mx = (int(cols[0]), int(cols[1]), int(cols[2]))
            except Exception:
                continue
            if mn <= 0 or opt <= 0 or mx <= 0:
                continue
            if not (mn <= opt <= mx):
                continue
            out.append((mn, opt, mx))
        return out

    def _choose_profile(self, total_tokens: int) -> int:
        n = int(total_tokens)
        if n <= 0:
            return 0
        if not self._profile_specs:
            return 0
            
        candidates: list[tuple[int, int]] = []
        for i, (mn, opt, mx) in enumerate(self._profile_specs):
            # Check if n is strictly within min/max bounds
            if int(mn) <= n <= int(mx):
                # Calculate distance to optimal
                dist = abs(n - int(opt))
                candidates.append((dist, int(i)))
                
        if not candidates:
            # Fallback: find profile that covers n with minimal violation or closest opt
            # If n > max of all profiles, pick the one with largest max
            # If n < min of all profiles, pick the one with smallest min
            # Here we just pick the profile whose opt is closest, but prefer one that contains n if possible
            # Re-scan to find 'closest' profile even if out of bounds
            best_i = 0
            min_dist = float('inf')
            for i, (mn, opt, mx) in enumerate(self._profile_specs):
                # Penalty for being out of bounds
                penalty = 0
                if n < mn: penalty = (mn - n) * 1000
                if n > mx: penalty = (n - mx) * 1000
                dist = abs(n - opt) + penalty
                if dist < min_dist:
                    min_dist = dist
                    best_i = i
            return int(best_i)
            
        candidates.sort() # Sort by distance to opt
        return int(candidates[0][1])

    def _set_profile(self, profile_idx: int, stream: int) -> None:
        idx = int(profile_idx)
        if idx == self._active_profile:
            return
        if idx < 0:
            idx = 0
        if idx >= self._num_profiles:
            idx = self._num_profiles - 1
            
        # Ensure we switch profile correctly
        try:
            if hasattr(self._context, "set_optimization_profile_async"):
                self._context.set_optimization_profile_async(int(idx), int(stream))
            elif hasattr(self._context, "set_optimization_profile"):
                self._context.set_optimization_profile(int(idx))
            else:
                 setattr(self._context, "active_optimization_profile", int(idx))
            self._active_profile = int(idx)
        except Exception:
            # If switching failed, we might be in a bad state, but try to continue
            pass

    def close(self) -> None:
        for buf in self._buffers.values():
            buf.free()
        self._buffers.clear()
        if self._shape_trace_on and self._shape_trace_dump:
            try:
                with open(self._shape_trace_dump, "w") as f:
                    json.dump(self._shape_trace_stats, f, indent=2, sort_keys=True)
            except Exception:
                pass
        destroy_stream(self._stream)

    def __enter__(self) -> "TensorRTRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def input_names(self) -> list[str]:
        return [n for n, t in self._tensors.items() if t.is_input]

    @property
    def output_names(self) -> list[str]:
        return [n for n, t in self._tensors.items() if not t.is_input]

    def _get_or_alloc(self, name: str, nbytes: int) -> DeviceBuffer:
        key = (name, int(nbytes))
        buf = self._buffers.get(key)
        if buf is not None and buf.nbytes >= nbytes and buf.ptr:
            return buf
        ptr = malloc(nbytes)
        buf = DeviceBuffer(ptr=ptr, nbytes=int(nbytes))
        self._buffers[key] = buf
        return buf

    def infer(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        use_torch_stream = bool(int(os.environ.get("LLMSCHEDULER_TRT_USE_TORCH_STREAM", "1")))
        if use_torch_stream:
            try:
                import torch
                stream = int(torch.cuda.current_stream().cuda_stream)
            except Exception:
                stream = int(self._stream)
        else:
            stream = int(self._stream)

        if "input_ids" in inputs:
            x = inputs["input_ids"]
            try:
                shape0 = int(getattr(x, "shape", (0,))[0])
            except Exception:
                shape0 = 0
            self._set_profile(self._choose_profile(shape0), int(stream))

        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

        for name in self.input_names:
            x = inputs[name]
            if hasattr(x, "data_ptr") and hasattr(x, "dtype"):
                import torch
                torch_dtype_map = {
                    torch.float32: np.float32,
                    torch.float16: np.float16,
                    torch.int32: np.int32,
                    torch.int64: np.int64,
                    torch.bool: np.bool_,
                    torch.int8: np.int8,
                }
                expected_dtype = self._tensors[name].dtype
                current_dtype = torch_dtype_map.get(x.dtype, None)
                if current_dtype != expected_dtype:
                    np_to_torch = {
                        np.dtype("float32"): torch.float32,
                        np.dtype("float16"): torch.float16,
                        np.dtype("int32"): torch.int32,
                        np.dtype("int64"): torch.int64,
                        np.dtype("bool"): torch.bool,
                        np.dtype("int8"): torch.int8,
                    }
                    target_torch_dtype = np_to_torch.get(np.dtype(expected_dtype), None)
                    if target_torch_dtype:
                        x = x.to(target_torch_dtype)
                    else:
                        x = x.cpu().numpy().astype(expected_dtype)

            torch_info = _torch_ptr_and_dtype(x)
            if torch_info is not None:
                ptr, _ = torch_info
                shape = tuple(int(d) for d in x.shape)
                self._context.set_input_shape(name, shape)
                self._context.set_tensor_address(name, ptr)
                continue

            arr = np.asarray(x)
            if arr.dtype != self._tensors[name].dtype:
                arr = arr.astype(self._tensors[name].dtype, copy=False)
            arr = np.ascontiguousarray(arr)
            shape = tuple(int(d) for d in arr.shape)
            self._context.set_input_shape(name, shape)
            buf = self._get_or_alloc(name, arr.nbytes)
            memcpy_htod_async(buf.ptr, arr, stream)
            self._context.set_tensor_address(name, buf.ptr)

        outputs: dict[str, np.ndarray] = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Output shape is dynamic but unresolved: {name} {shape}")
            dt = self._tensors[name].dtype
            out = np.empty(shape, dtype=dt)
            buf = self._get_or_alloc(name, out.nbytes)
            self._context.set_tensor_address(name, buf.ptr)
            outputs[name] = out

        ok = self._context.execute_async_v3(self._stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        for name, out in outputs.items():
            buf = self._get_or_alloc(name, out.nbytes)
            memcpy_dtoh_async(out, buf.ptr, stream)

        stream_synchronize(stream)
        return outputs

    def infer_torch(self, inputs: dict[str, Any]):
        import torch
        use_torch_stream = bool(int(os.environ.get("LLMSCHEDULER_TRT_USE_TORCH_STREAM", "1")))
        use_dedicated = use_torch_stream and bool(int(os.environ.get("LLMSCHEDULER_TRT_DEDICATED_STREAM", "1")))

        cur = torch.cuda.current_stream()
        if use_dedicated:
            s = getattr(self, "_torch_stream", None)
            if s is None:
                s = torch.cuda.Stream()
                self._torch_stream = s
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                outputs = self._infer_torch_on_stream(inputs, torch, int(torch.cuda.current_stream().cuda_stream), use_torch_stream)
            cur.wait_stream(s)
            return outputs

        return self._infer_torch_on_stream(inputs, torch, int(cur.cuda_stream), use_torch_stream)

    def _infer_torch_on_stream_cudagraph(self, inputs: dict[str, Any], torch, stream: int):
        if bool(getattr(self, "_cudagraph_disable_all", False)):
            return self._infer_torch_on_stream_torch_baseline(inputs, torch, stream)
        if "input_ids" in inputs:
            x0 = inputs["input_ids"]
            try:
                shape0 = int(getattr(x0, "shape", (0,))[0])
            except Exception:
                shape0 = 0
            self._set_profile(self._choose_profile(shape0), int(stream))

        if bool(int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_FALLBACK", "1"))):
            try:
                return self._infer_torch_on_stream_cudagraph_impl(inputs, torch, stream)
            except Exception:
                failures = int(getattr(self, "_cudagraph_failures", 0)) + 1
                self._cudagraph_failures = failures
                disable_after = int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_DISABLE_AFTER", "3") or 3)
                if failures >= disable_after:
                    self._cudagraph_disable_all = True
                return self._infer_torch_on_stream_torch_baseline(inputs, torch, stream)
        return self._infer_torch_on_stream_cudagraph_impl(inputs, torch, stream)

    def _infer_torch_on_stream_torch_baseline(self, inputs: dict[str, Any], torch, stream: int):
        in_shapes: dict[str, tuple[int, ...]] = {}
        in_tensors: dict[str, Any] = {}
        for name in self.input_names:
            x = inputs.get(name)
            if not isinstance(x, torch.Tensor):
                raise ValueError("Torch baseline requires torch.Tensor inputs")
            if not x.is_cuda:
                x = x.to("cuda")
            if not x.is_contiguous():
                x = x.contiguous()
            in_tensors[name] = x
            in_shapes[name] = tuple(int(d) for d in x.shape)
        for name in self.input_names:
            self._context.set_input_shape(name, in_shapes[name])

        out_shapes: dict[str, tuple[int, ...]] = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Output shape is dynamic but unresolved: {name} {shape}")
            out_shapes[name] = shape

        torch_dtype_map = {
            np.dtype("float32"): torch.float32,
            np.dtype("float16"): torch.float16,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
            np.dtype("bool"): torch.bool,
            np.dtype("int8"): torch.int8,
        }

        for n in self.input_names:
            self._context.set_tensor_address(n, int(in_tensors[n].data_ptr()))

        outputs: dict[str, Any] = {}
        for n in self.output_names:
            dt = self._tensors[n].dtype
            torch_dt = torch_dtype_map.get(np.dtype(dt), None)
            if torch_dt is None:
                raise RuntimeError(f"Unsupported output dtype for torch: {n} {dt}")
            needed_numel = int(np.prod(out_shapes[n], dtype=np.int64))
            buf = self._torch_output_buffers.get(n)
            if buf is None or buf.dtype != torch_dt or int(buf.numel()) < needed_numel:
                buf = torch.empty((needed_numel,), dtype=torch_dt, device="cuda")
                self._torch_output_buffers[n] = buf
            out = buf[:needed_numel].view(out_shapes[n])
            self._context.set_tensor_address(n, int(out.data_ptr()))
            outputs[n] = out

        ok = self._context.execute_async_v3(int(stream))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed (baseline)")
        return outputs

    def _infer_torch_on_stream_cudagraph_impl(self, inputs: dict[str, Any], torch, stream: int):
        in_shapes: dict[str, tuple[int, ...]] = {}
        in_tensors: dict[str, Any] = {}
        for name in self.input_names:
            x = inputs.get(name)
            if not isinstance(x, torch.Tensor):
                raise ValueError("CUDA graph mode requires torch.Tensor inputs")
            if not x.is_cuda:
                x = x.to("cuda")
            if not x.is_contiguous():
                x = x.contiguous()
            in_tensors[name] = x
            in_shapes[name] = tuple(int(d) for d in x.shape)
        for name in self.input_names:
            self._context.set_input_shape(name, in_shapes[name])

        out_shapes: dict[str, tuple[int, ...]] = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Output shape is dynamic but unresolved: {name} {shape}")
            out_shapes[name] = shape

        ctx_key_mode = str(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_CTX_KEY", "") or "").strip().lower()
        ctx_key = None
        if ctx_key_mode and self._ctx is not None:
            try:
                req_ids = tuple(int(x) for x in getattr(self._ctx, "current_batch_req_ids", []) or [])
                total_lens = [int(x) for x in getattr(self._ctx, "current_batch_total_lens", []) or []]
                page_size = int(getattr(self._ctx, "page_size", 0) or 0)
                if ctx_key_mode == "total_len":
                    ctx_key = (req_ids, tuple(total_lens))
                elif ctx_key_mode in ("page_and_last", "page_last"):
                    if page_size > 0 and total_lens:
                        pages = tuple(int(t) // int(page_size) for t in total_lens)
                        last_bucket = int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_LAST_BUCKET", "0") or 0)
                        if last_bucket > 0:
                            last = tuple(int((int(t) % int(page_size)) // int(last_bucket)) for t in total_lens)
                        else:
                            last = tuple(int(t) % int(page_size) for t in total_lens)
                        ctx_key = (req_ids, pages, last)
            except Exception:
                ctx_key = None

        key = (
            int(stream),
            str(ctx_key_mode),
            ctx_key,
            tuple((n, in_shapes[n], str(self._tensors[n].dtype)) for n in self.input_names),
            tuple((n, out_shapes[n], str(self._tensors[n].dtype)) for n in self.output_names),
        )
        cache = getattr(self, "_cudagraph_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._cudagraph_cache = cache
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict(cache)
            self._cudagraph_cache = cache

        disabled_keys = getattr(self, "_cudagraph_disabled_keys", None)
        if disabled_keys is None:
            disabled_keys = set()
            self._cudagraph_disabled_keys = disabled_keys
        if key in disabled_keys:
            return self._infer_torch_on_stream_torch_baseline(inputs, torch, stream)

        entry = cache.get(key)
        if entry is not None:
            cache.move_to_end(key)

        if entry is None:
            if bool(int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_SINGLETON", "1"))):
                cache.clear()
            exec_ctx = self._context
            if (not bool(int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_SINGLETON", "1")))) and bool(
                int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_MULTI_EXEC_CTX", "1"))
            ):
                exec_ctx = self._engine.create_execution_context()
                if exec_ctx is None:
                    raise RuntimeError("Failed to create TensorRT execution context (OOM)")
                try:
                    if hasattr(exec_ctx, "set_optimization_profile_async"):
                        exec_ctx.set_optimization_profile_async(int(self._active_profile), int(stream))
                    elif hasattr(exec_ctx, "set_optimization_profile"):
                        exec_ctx.set_optimization_profile(int(self._active_profile))
                    else:
                        setattr(exec_ctx, "active_optimization_profile", int(self._active_profile))
                except Exception:
                    pass
                for name in self.input_names:
                    exec_ctx.set_input_shape(name, in_shapes[name])
                out_shapes = {}
                for name in self.output_names:
                    shape = tuple(int(d) for d in exec_ctx.get_tensor_shape(name))
                    if any(d < 0 for d in shape):
                        raise RuntimeError(f"Output shape is dynamic but unresolved: {name} {shape}")
                    out_shapes[name] = shape
            static_inputs: dict[str, Any] = {}
            for n in self.input_names:
                x = in_tensors[n]
                static_inputs[n] = torch.empty_like(x)
                exec_ctx.set_tensor_address(n, int(static_inputs[n].data_ptr()))

            torch_dtype_map = {
                np.dtype("float32"): torch.float32,
                np.dtype("float16"): torch.float16,
                np.dtype("int32"): torch.int32,
                np.dtype("int64"): torch.int64,
                np.dtype("bool"): torch.bool,
                np.dtype("int8"): torch.int8,
            }
            static_outputs: dict[str, Any] = {}
            for n in self.output_names:
                dt = self._tensors[n].dtype
                torch_dt = torch_dtype_map.get(np.dtype(dt), None)
                if torch_dt is None:
                    raise RuntimeError(f"Unsupported output dtype for torch: {n} {dt}")
                needed_numel = int(np.prod(out_shapes[n], dtype=np.int64))
                buf = self._torch_output_buffers.get(n)
                if buf is None or buf.dtype != torch_dt or int(buf.numel()) < needed_numel:
                    buf = torch.empty((needed_numel,), dtype=torch_dt, device="cuda")
                    self._torch_output_buffers[n] = buf
                out = buf[:needed_numel].view(out_shapes[n])
                exec_ctx.set_tensor_address(n, int(out.data_ptr()))
                static_outputs[n] = out

            for n in self.input_names:
                static_inputs[n].copy_(in_tensors[n])

            ok = exec_ctx.execute_async_v3(stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed (warmup)")
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            cap_stream = torch.cuda.current_stream()
            try:
                if int(getattr(cap_stream, "cuda_stream", 0) or 0) != int(stream):
                    cap_stream = torch.cuda.ExternalStream(int(stream))
            except Exception:
                cap_stream = torch.cuda.current_stream()
            with torch.cuda.stream(cap_stream):
                try:
                    with torch.cuda.graph(g, stream=cap_stream):
                        ok = exec_ctx.execute_async_v3(stream)
                        if not ok:
                            raise RuntimeError("TensorRT execute_async_v3 failed (capture)")
                except TypeError:
                    with torch.cuda.graph(g):
                        ok = exec_ctx.execute_async_v3(stream)
                        if not ok:
                            raise RuntimeError("TensorRT execute_async_v3 failed (capture)")
            entry = {"graph": g, "inputs": static_inputs, "outputs": static_outputs}
            if exec_ctx is not self._context:
                entry["exec_ctx"] = exec_ctx
            cache[key] = entry
            cache.move_to_end(key)
            max_graphs = int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_MAX_GRAPHS", "1") or 1)
            while len(cache) > max_graphs and max_graphs > 0:
                old_key, _ = cache.popitem(last=False)
                try:
                    disabled_keys.discard(old_key)
                except Exception:
                    pass

            if bool(int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_CHECK", "0"))):
                tol = float(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH_CHECK_TOL", "0.01") or 0.01)
                base_out = self._infer_torch_on_stream_torch_baseline(inputs, torch, stream)
                static_inputs = entry["inputs"]
                cap_stream2 = torch.cuda.current_stream()
                try:
                    if int(getattr(cap_stream2, "cuda_stream", 0) or 0) != int(stream):
                        cap_stream2 = torch.cuda.ExternalStream(int(stream))
                except Exception:
                    cap_stream2 = torch.cuda.current_stream()
                with torch.cuda.stream(cap_stream2):
                    for n in self.input_names:
                        static_inputs[n].copy_(in_tensors[n])
                    entry["graph"].replay()
                torch.cuda.synchronize()
                ok_check = True
                for n in self.output_names:
                    try:
                        diff = (base_out[n] - entry["outputs"][n]).abs().max().item()
                    except Exception:
                        diff = float("inf")
                    if not (diff <= tol):
                        ok_check = False
                        break
                if not ok_check:
                    try:
                        cache.pop(key, None)
                    except Exception:
                        pass
                    disabled_keys.add(key)
                    return base_out

        static_inputs = entry["inputs"]
        cap_stream = torch.cuda.current_stream()
        try:
            if int(getattr(cap_stream, "cuda_stream", 0) or 0) != int(stream):
                cap_stream = torch.cuda.ExternalStream(int(stream))
        except Exception:
            cap_stream = torch.cuda.current_stream()
        with torch.cuda.stream(cap_stream):
            for n in self.input_names:
                static_inputs[n].copy_(in_tensors[n])
            entry["graph"].replay()
        return dict(entry["outputs"])

    def _infer_torch_on_stream(self, inputs: dict[str, Any], torch, stream: int, use_torch_stream: bool):
        try:
            import orchid.llmscheduler.plugins.composite_attention as ca
            cur_ctx = getattr(ca, "CURRENT_CTX", None)
            if cur_ctx is not None:
                ca.set_context(cur_ctx, stream=stream)
        except Exception:
            pass

        if "input_ids" in inputs:
            x0 = inputs["input_ids"]
            try:
                shape0 = int(getattr(x0, "shape", (0,))[0])
            except Exception:
                shape0 = 0
            self._set_profile(self._choose_profile(shape0), int(stream))

        if bool(int(os.environ.get("LLMSCHEDULER_TRT_CUDAGRAPH", "0"))):
            return self._infer_torch_on_stream_cudagraph(inputs, torch, stream)

        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

        torch_dtype_map = {
            np.dtype("float32"): torch.float32,
            np.dtype("float16"): torch.float16,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
            np.dtype("bool"): torch.bool,
            np.dtype("int8"): torch.int8,
        }

        for name in self.input_names:
            x = inputs[name]
            if isinstance(x, torch.Tensor):
                expected_dtype = self._tensors[name].dtype
                target_torch_dtype = torch_dtype_map.get(np.dtype(expected_dtype), None)
                if target_torch_dtype is not None and x.dtype != target_torch_dtype:
                    x = x.to(target_torch_dtype)
                if not x.is_cuda:
                    x = x.to("cuda")
                if not x.is_contiguous():
                    x = x.contiguous()
                shape = tuple(int(d) for d in x.shape)
                if self._shape_trace_on:
                    st = self._shape_trace_stats.get(name)
                    if st is None:
                        st = {"count": 0, "unique": 0, "min": None, "max": None, "hist": {}}
                        self._shape_trace_stats[name] = st
                    st["count"] = int(st["count"]) + 1
                    hist = st["hist"]
                    key = str(shape)
                    hist[key] = int(hist.get(key, 0)) + 1
                    st["unique"] = int(len(hist))
                    if st["min"] is None or int(shape[0]) < int(st["min"][0]):
                        st["min"] = list(shape)
                    if st["max"] is None or int(shape[0]) > int(st["max"][0]):
                        st["max"] = list(shape)
                self._context.set_input_shape(name, shape)
                self._context.set_tensor_address(name, int(x.data_ptr()))
                continue

            arr = np.asarray(x)
            if arr.dtype != self._tensors[name].dtype:
                arr = arr.astype(self._tensors[name].dtype, copy=False)
            arr = np.ascontiguousarray(arr)
            shape = tuple(int(d) for d in arr.shape)
            if self._shape_trace_on:
                st = self._shape_trace_stats.get(name)
                if st is None:
                    st = {"count": 0, "unique": 0, "min": None, "max": None, "hist": {}}
                    self._shape_trace_stats[name] = st
                st["count"] = int(st["count"]) + 1
                hist = st["hist"]
                key = str(shape)
                hist[key] = int(hist.get(key, 0)) + 1
                st["unique"] = int(len(hist))
                if st["min"] is None or int(shape[0]) < int(st["min"][0]):
                    st["min"] = list(shape)
                if st["max"] is None or int(shape[0]) > int(st["max"][0]):
                    st["max"] = list(shape)
            self._context.set_input_shape(name, shape)
            buf = self._get_or_alloc(name, arr.nbytes)
            memcpy_htod_async(buf.ptr, arr, stream)
            self._context.set_tensor_address(name, buf.ptr)

        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"Output shape is dynamic but unresolved: {name} {shape}")
            dt = self._tensors[name].dtype
            torch_dt = torch_dtype_map.get(np.dtype(dt), None)
            if torch_dt is None:
                raise RuntimeError(f"Unsupported output dtype for torch: {name} {dt}")
            needed_numel = int(np.prod(shape, dtype=np.int64))
            buf = self._torch_output_buffers.get(name)
            if buf is None or buf.dtype != torch_dt or int(buf.numel()) < needed_numel:
                buf = torch.empty((needed_numel,), dtype=torch_dt, device="cuda")
                self._torch_output_buffers[name] = buf
            out = buf[:needed_numel].view(shape)
            self._context.set_tensor_address(name, int(out.data_ptr()))
            outputs[name] = out

        if self._layer_profile_on and self._layer_profiler is not None:
            try:
                self._layer_profiler.reset()
            except Exception:
                pass

        ok = self._context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        if not use_torch_stream:
            stream_synchronize(stream)
        if self._shape_trace_on:
            self._shape_trace_step += 1
            if self._shape_trace_interval > 0 and (self._shape_trace_step % self._shape_trace_interval) == 0:
                for n, st in self._shape_trace_stats.items():
                    hist = st.get("hist", {})
                    items = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[: int(self._shape_trace_topk)]
                    top = ", ".join([f"{k}:{v}" for k, v in items])
                    print(
                        f"[trt_shape] name={n} count={int(st.get('count', 0))} unique={int(st.get('unique', 0))} min={st.get('min')} max={st.get('max')} top={top}"
                    )
                if self._shape_trace_dump:
                    try:
                        with open(self._shape_trace_dump, "w") as f:
                            json.dump(self._shape_trace_stats, f, indent=2, sort_keys=True)
                    except Exception:
                        pass
        if self._layer_profile_on and self._layer_profiler is not None:
            self._layer_profile_step += 1
            do_print = self._layer_profile_interval > 0 and (self._layer_profile_step % self._layer_profile_interval) == 0
            do_dump = bool(self._layer_profile_dump)
            if do_print or do_dump:
                items = list(getattr(self._layer_profiler, "times", {}).items())
                items.sort(key=lambda kv: kv[1], reverse=True)
                topk = int(max(1, self._layer_profile_topk))
                payload = {
                    "step": int(self._layer_profile_step),
                    "topk": [{"name": k, "ms": float(v)} for k, v in items[:topk]],
                    "layers": len(items),
                }
                if do_print:
                    top = ", ".join([f"{x['name']}:{x['ms']:.3f}" for x in payload["topk"]])
                    print(f"[trt_layer] step={payload['step']} layers={payload['layers']} top={top}", flush=True)
                if do_dump:
                    try:
                        with open(self._layer_profile_dump, "a") as f:
                            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
        return outputs
