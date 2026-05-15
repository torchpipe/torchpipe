import ctypes
import os
import threading

import numpy as np
import torch
import tvm_ffi
from tvm_ffi import cpp

_PAGE_MANAGER_CDLL = None
_SCHEDULER_PLAN_MOD = None
_SCHEDULER_PLAN_LOCK = threading.Lock()


def ensure_scheduler_plan_ops_loaded() -> None:
    global _SCHEDULER_PLAN_MOD
    if _SCHEDULER_PLAN_MOD is not None:
        return
    with _SCHEDULER_PLAN_LOCK:
        if _SCHEDULER_PLAN_MOD is not None:
            return
        f1 = tvm_ffi.get_global_func("custom.schedule_step_plan", allow_missing=True)
        f2 = tvm_ffi.get_global_func("custom.pack_padded_prefix_i64", allow_missing=True)
        if f1 is not None and f2 is not None:
            _SCHEDULER_PLAN_MOD = True
            return
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cpp_plugin_dir = os.path.join(current_dir, "..", "plugins", "csrc")
        lib_path = cpp.build(
            name="orchid_scheduler_plan_only",
            cpp_files=[os.path.join(cpp_plugin_dir, "scheduler_plan.cc")],
            extra_include_paths=[cpp_plugin_dir],
            extra_cflags=["-std=c++17", "-O3"],
        )
        _SCHEDULER_PLAN_MOD = tvm_ffi.load_module(lib_path)

class CppPageManager:
    def __init__(self, max_pages=4096):
        global _PAGE_MANAGER_CDLL
        f_init = tvm_ffi.get_global_func("custom.init", allow_missing=True)
        f_sched = tvm_ffi.get_global_func("custom.schedule_step", allow_missing=True)
        if f_init is None or f_sched is None:
            # Locate C++ source files
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # v11/llmscheduler/core/allocator.py -> v11/llmscheduler/plugins/csrc/
            cpp_plugin_dir = os.path.join(current_dir, "..", "plugins", "csrc")
            include_dir = os.path.join(current_dir, "..", "plugins", "include")
            
            sources = [
                os.path.join(cpp_plugin_dir, "page_manager.cc"),
                os.path.join(cpp_plugin_dir, "batch_utils.cc"),
                os.path.join(cpp_plugin_dir, "binding.cc"),
            ]
            if f_sched is None:
                sources.append(os.path.join(cpp_plugin_dir, "scheduler.cc"))
            
            # JIT Compilation
            try:
                lib_path = cpp.build(
                    name="page_manager_jit",
                    cpp_files=sources,
                    extra_include_paths=[cpp_plugin_dir, include_dir],
                    extra_cflags=["-std=c++17", "-O3"]
                )
                _PAGE_MANAGER_CDLL = tvm_ffi.load_module(lib_path)
                
                # Debug: list all functions if possible or just try to force load
                # Sometimes static init blocks need explicit symbol reference or specific linker flags?
                # But default flags include -shared.
                
            except Exception as e:
                raise RuntimeError(f"JIT Compilation failed: {e}")
        self._cdll = _PAGE_MANAGER_CDLL
        self.mod = _PAGE_MANAGER_CDLL

        # The C++ code uses TVM_FFI_STATIC_INIT_BLOCK to register functions globally.
        # So we should use tvm_ffi.get_global_func instead of self.mod.get_function
        # if the module object doesn't automatically wrap global registry for that lib.
        try:
            self.init_func = tvm_ffi.get_global_func("custom.init")
            self.get_pages_func = tvm_ffi.get_global_func("custom.get_pages")
            self.free_func = tvm_ffi.get_global_func("custom.free")
            self.reset_func = tvm_ffi.get_global_func("custom.reset")
        except Exception as e:
            # Fallback or re-raise
            print(f"Global function lookup failed: {e}. Trying module lookup...")
            self.init_func = self.mod.get_function("custom.init")
            self.get_pages_func = self.mod.get_function("custom.get_pages")
            self.free_func = self.mod.get_function("custom.free")
            self.reset_func = self.mod.get_function("custom.reset")
        
        try:
            self.prepare_step_func = tvm_ffi.get_global_func("custom.prepare_step")
        except:
            self.prepare_step_func = None
            print("Warning: custom.prepare_step not found in library.")
        
        self.init_func(max_pages)
        self.max_pages = max_pages
        
    def get_pages(self, req_id, num_needed):
        # Legacy function if needed
        pages_list = self.get_pages_func(req_id, num_needed)
        out_np = np.array(pages_list, dtype=np.int32)
        return torch.from_numpy(out_np).to("cuda")

    def free(self, req_id):
        self.free_func(req_id)
        
    def reset(self):
        self.reset_func()

    def close(self):
        # Explicitly release references to TVM functions
        self.init_func = None
        self.get_pages_func = None
        self.free_func = None
        self.reset_func = None
        self.prepare_step_func = None
        self.mod = None
        self._cdll = None
