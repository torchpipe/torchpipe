from abc import ABC, abstractmethod
import torch
import numpy as np
import tvm_ffi
from .allocator import CppPageManager

class BatchMetadata:
    def __init__(self, indptr, indices, last_page_len, qo_indptr, slot_mapping, batch_indices=None, positions=None):
        self.indptr = indptr
        self.indices = indices
        self.last_page_len = last_page_len
        self.qo_indptr = qo_indptr
        self.slot_mapping = slot_mapping
        self.batch_indices = batch_indices
        self.positions = positions

class MetadataBuilder(ABC):
    @abstractmethod
    def prepare_step(self, req_ids, total_lens, new_tokens, page_size, layer_idx, num_layers):
        pass

class PythonMetadataBuilder(MetadataBuilder):
    def __init__(self, page_manager: CppPageManager, device="cuda"):
        self.pm = page_manager
        self.device = device
        
    def prepare_step(self, req_ids, total_lens, new_tokens, page_size, layer_idx, num_layers):
        indptr = [0]
        indices = []
        last_page_len = []
        qo_indptr = [0]
        slot_mapping = []
        batch_indices = []
        positions = []
        
        current_page_offset = 0
        current_qo_offset = 0
        
        for i, req_id in enumerate(req_ids):
            total_len = total_lens[i]
            new_len = new_tokens[i]
            
            unique_id = req_id
            num_pages = (total_len + page_size - 1) // page_size
            
            pages_tensor = self.pm.get_pages(unique_id, num_pages)
            pages_list = pages_tensor.cpu().numpy().tolist()
            
            indices.extend(pages_list)
            current_page_offset += len(pages_list)
            indptr.append(current_page_offset)
            
            last_len = (total_len - 1) % page_size + 1
            last_page_len.append(last_len)
            
            current_qo_offset += new_len
            qo_indptr.append(current_qo_offset)
            
            offset = total_len - new_len
            for t in range(new_len):
                abs_pos = offset + t
                page_idx = abs_pos // page_size
                page_offset = abs_pos % page_size
                
                physical_page = pages_list[page_idx]
                slot = physical_page * page_size + page_offset
                slot_mapping.append(slot)
                batch_indices.append(i)
                positions.append(abs_pos)
                
        return BatchMetadata(
            torch.tensor(indptr, dtype=torch.int32, device=self.device),
            torch.tensor(indices, dtype=torch.int32, device=self.device),
            torch.tensor(last_page_len, dtype=torch.int32, device=self.device),
            torch.tensor(qo_indptr, dtype=torch.int32, device=self.device),
            torch.tensor(slot_mapping, dtype=torch.int32, device=self.device),
            torch.tensor(batch_indices, dtype=torch.int32, device=self.device),
            torch.tensor(positions, dtype=torch.int32, device=self.device),
        )

class CppMetadataBuilder(MetadataBuilder):
    def __init__(self, page_manager: CppPageManager, device="cuda"):
        self.pm = page_manager
        self.device = device
        self._cuda_bufs = {}
        
    def prepare_step(self, req_ids, total_lens, new_tokens, page_size, layer_idx, num_layers):
        if self.pm.prepare_step_func is None:
            raise RuntimeError("C++ Library does not support prepare_step")

        def _to_cpu_int32_tensor(x):
            if isinstance(x, torch.Tensor):
                if x.device.type != "cpu":
                    x = x.to("cpu")
                if x.dtype != torch.int32:
                    x = x.to(torch.int32)
                if not x.is_contiguous():
                    x = x.contiguous()
                return x
            return torch.as_tensor(x, dtype=torch.int32, device="cpu")
        
        req_ids_tensor = _to_cpu_int32_tensor(req_ids)
        total_lens_tensor = _to_cpu_int32_tensor(total_lens)
        new_tokens_tensor = _to_cpu_int32_tensor(new_tokens)
        
        ret = self.pm.prepare_step_func(
            req_ids_tensor, 
            total_lens_tensor, 
            new_tokens_tensor, 
            page_size, 
            layer_idx, 
            num_layers
        )
        
        def _copy_to_cuda(name: str, t: torch.Tensor) -> torch.Tensor:
            if t.device.type == "cuda":
                if t.dtype != torch.int32:
                    if t.dtype.is_floating_point:
                        t = t.to(dtype=torch.int64)
                    t = t.to(dtype=torch.int32)
                if not t.is_contiguous():
                    t = t.contiguous()
                return t
            if t.device.type != "cpu":
                t = t.to("cpu")
            if t.dtype != torch.int32:
                if t.dtype.is_floating_point:
                    t = t.to(dtype=torch.int64)
                t = t.to(dtype=torch.int32)
            if not t.is_contiguous():
                t = t.contiguous()
            flat = t.view(-1)
            buf = self._cuda_bufs.get(name)
            if buf is None or int(buf.numel()) < int(flat.numel()):
                buf = torch.empty((int(flat.numel()),), device=self.device, dtype=torch.int32)
                self._cuda_bufs[name] = buf
            out = buf[: int(flat.numel())].view(t.shape)
            out.copy_(t)
            return out

        def tvm_to_torch(name: str, tvm_tensor):
            dlpack = tvm_tensor.__dlpack__()
            t = torch.from_dlpack(dlpack)
            return _copy_to_cuda(name, t)
        
        indptr = tvm_to_torch("indptr", ret[0])
        indices = tvm_to_torch("indices", ret[1])
        last_page_len = tvm_to_torch("last_page_len", ret[2])
        qo_indptr = tvm_to_torch("qo_indptr", ret[3])
        slot_mapping = tvm_to_torch("slot_mapping", ret[4])
        batch_indices = tvm_to_torch("batch_indices", ret[5]) if len(ret) > 5 else None
        positions = tvm_to_torch("positions", ret[6]) if len(ret) > 6 else None
        
        return BatchMetadata(indptr, indices, last_page_len, qo_indptr, slot_mapping, batch_indices, positions)
