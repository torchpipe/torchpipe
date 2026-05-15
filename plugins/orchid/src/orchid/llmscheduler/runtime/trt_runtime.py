from .base import ModelRuntime, AttentionContext
from orchid.llmscheduler.plugins.composite_attention import set_context
import os
import tensorrt as trt
import torch

class TensorRTModelRuntime(ModelRuntime):
    def __init__(self, model_path, use_fp16=False, engine_path: str | None = None):
        from orchid.llmscheduler.trt.builder import EngineBuilder
        
        # Load ONNX and Build TRT Engine
        self.builder = EngineBuilder()
        
        spec = os.environ.get("LLMSCHEDULER_TRT_INPUT_IDS_PROFILES", "").strip()
        if not spec:
            spec = "1,10,32;32,512,1024;1024,4096,8192;8192,16384,40960"

        profiles = []
        for part in spec.split(";"):
            cols = [c.strip() for c in part.split(",")]
            if len(cols) != 3:
                continue
            mn, opt, mx = (int(cols[0]), int(cols[1]), int(cols[2]))
            profiles.append({"input_ids": [(mn,), (opt,), (mx,)]})
        if not profiles:
            profiles = [{"input_ids": [(1,), (10,), (40960,)]}]
        
        self.engine_bytes = self.builder.build(
            model_path,
            fp16=use_fp16,
            input_profile=profiles,
            verbose=False,
            engine_path=engine_path,
        )
        self.runtime = None 
        
    def init_runtime(self, ctx: AttentionContext):
        from orchid.llmscheduler.trt.runtime import TensorRTRuntime as TRTRuntime
        self.runtime = TRTRuntime(engine_bytes=self.engine_bytes, ctx=ctx)
        os.environ.setdefault("LLMSCHEDULER_TRT_USE_TORCH_STREAM", "1")

    def forward(self, input_tensor: torch.Tensor, ctx: AttentionContext) -> torch.Tensor:
        if self.runtime is None:
            self.init_runtime(ctx)
            
        set_context(ctx)
        outputs = self.runtime.infer_torch({"input_ids": input_tensor})
        return outputs["logits"]
