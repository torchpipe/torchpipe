import torch
import numpy as np
import os
from abc import ABC, abstractmethod

class AttentionContext:
    def __init__(self, num_layers, num_heads, kv_num_heads, head_dim, page_size, max_pages, use_cpp_metadata: bool, device="cuda", use_fp16: bool = False):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.device = device
        
        # KV Cache - FlashInfer supports float16/bfloat16 kernels; keep cache in float16
        pages_per_layer = int(max(1, int(max_pages) // int(max(1, int(num_layers)))))
        self.pages_per_layer = pages_per_layer
        self.kv_cache = torch.zeros(
            int(num_layers),
            pages_per_layer,
            2,
            int(page_size),
            int(kv_num_heads),
            int(head_dim),
            dtype=torch.float16,
            device=device,
        )
        
        # FlashInfer Wrappers
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
        import flashinfer
        workspace_mb = int(os.environ.get("LLMSCHEDULER_FLASHINFER_WORKSPACE_MB", "512"))
        workspace_bytes = int(workspace_mb) * 1024 * 1024
        workspace_buffer_prefill = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        workspace_buffer_prefill_ragged = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        workspace_buffer_decode = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        use_tensor_cores = bool(int(os.environ.get("LLMSCHEDULER_FLASHINFER_USE_TENSOR_CORES", "0")))
        self.prefill_paged_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer_prefill, "NHD")
        self.prefill_ragged_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer_prefill_ragged, "NHD")
        if bool(int(os.environ.get("LLMSCHEDULER_FLASHINFER_DECODE_CUDAGRAPH", "0"))):
            max_bs = int(os.environ.get("LLMSCHEDULER_FLASHINFER_MAX_BATCH_SIZE", "256"))
            max_bs = max(1, int(max_bs))
            indptr_buffer = torch.empty((int(max_bs) + 1,), dtype=torch.int32, device=device)
            last_page_len_buffer = torch.empty((int(max_bs),), dtype=torch.int32, device=device)
            indices_buffer = torch.empty((int(pages_per_layer),), dtype=torch.int32, device=device)
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer_decode,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=bool(use_tensor_cores),
                paged_kv_indptr_buffer=indptr_buffer,
                paged_kv_indices_buffer=indices_buffer,
                paged_kv_last_page_len_buffer=last_page_len_buffer,
            )
            self._decode_cudagraph_buffers = {
                "indptr": indptr_buffer,
                "indices": indices_buffer,
                "last_page_len": last_page_len_buffer,
                "max_bs": int(max_bs),
            }
        else:
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer_decode, "NHD", use_tensor_cores=bool(use_tensor_cores))
        
        # Metadata
        self.metadata = None
        self.current_batch_req_ids = []
        self.current_batch_input_ids = []
        self.current_batch_seq_lens = []
        self.current_batch_total_lens = []
        self.current_batch_history_lens = []
        self.current_batch_is_prefill = []
        self.is_all_decode = False
        
        # Page Manager & Metadata Builder
        from ..core.allocator import CppPageManager
        from ..core.metadata import CppMetadataBuilder, PythonMetadataBuilder
        
        assert use_cpp_metadata, "PythonMetadataBuilder is not supported yet"
        self.page_manager = CppPageManager(max_pages=pages_per_layer)
        self.mb = CppMetadataBuilder(self.page_manager, device=device) if use_cpp_metadata else PythonMetadataBuilder(self.page_manager, device=device)

    def close(self):
        if hasattr(self, "page_manager") and self.page_manager:
            self.page_manager.close()

class ModelRuntime(ABC):
    @abstractmethod
    def forward(self, input_tensor: torch.Tensor, ctx: AttentionContext) -> torch.Tensor:
        """
        Executes the model forward pass.
        
        Args:
            input_tensor: Input IDs tensor [batch_size, seq_len] or [total_tokens]
            ctx: AttentionContext containing current batch metadata and state
            
        Returns:
            Logits tensor
        """
        pass
