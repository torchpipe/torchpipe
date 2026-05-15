import tvm_ffi
from tvm_ffi import cpp
import numpy as np
import os
import ctypes

# Load the custom library
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the library using tvm_ffi.cpp.build to ensure it exists and is up to date
cpp_plugin_dir = current_dir
sources = [
    os.path.join(cpp_plugin_dir, "page_manager.cc"),
    os.path.join(cpp_plugin_dir, "batch_utils.cc"),
    os.path.join(cpp_plugin_dir, "binding.cc")
]
include_dir = os.path.join(current_dir, "..", "include") # Assuming include is parallel to csrc?
# Check where include dir is.
# In allocator.py: include_dir = os.path.join(current_dir, "..", "plugins", "include")
# Here current_dir is plugins/csrc. So .. is plugins. .. / include is plugins/include.
# Wait, allocator.py is in core/. current_dir is core/.
# So .. is llmscheduler/. .. / plugins / include is llmscheduler/plugins/include.
# Here current_dir is llmscheduler/plugins/csrc.
# So .. is llmscheduler/plugins.
# So ../include is llmscheduler/plugins/include.
include_dir = os.path.join(current_dir, "..", "include")

try:
    lib_path = cpp.build(
        name="page_manager_test_jit",
        cpp_files=sources,
        extra_include_paths=[cpp_plugin_dir, include_dir],
        extra_cflags=["-std=c++17", "-O3"]
    )
    mod = tvm_ffi.load_module(lib_path)
except Exception as e:
    print(f"Compilation failed: {e}")
    # Fallback to loading local if exists, though unlikely if build failed
    lib_path = os.path.join(current_dir, "libpage_manager.so")
    if os.path.exists(lib_path):
        mod = tvm_ffi.load_module(lib_path)
    else:
        raise e


# Wrapper Class
class PageManager:
    def __init__(self, max_pages=4096):
        # We use global functions for now, as the C++ implementation uses a global instance.
        # Ideally, we should register an Object type, but global functions are simpler for this demo.
        # We call init to set up the global instance.
        
        # Note: tvm_ffi.get_global_func raises if not found.
        # We need to make sure the library is loaded and symbols are registered.
        # load_module usually registers functions if they are exported via TVM_REGISTER_GLOBAL.
        
        self.init_func = tvm_ffi.get_global_func("custom.init")
        self.get_pages_func = tvm_ffi.get_global_func("custom.get_pages")
        self.free_func = tvm_ffi.get_global_func("custom.free")
        self.reset_func = tvm_ffi.get_global_func("custom.reset")
        
        self.init_func(max_pages)
        
    def get_pages(self, req_id, num_needed):
        # Prepare output buffer
        # In real scenario, we might want to avoid allocation every time.
        # Here we allocate a numpy array big enough.
        # Actually, we don't know exactly how many pages we *will* get if we just requested "needed"?
        # Wait, the C++ logic returns *existing* + *new* = total pages.
        # So we expect to get `num_needed` pages if successful.
        
        out_np = np.zeros(num_needed, dtype=np.int32)
        
        # Call FFI
        # Note: tvm_ffi automatically converts numpy array to DLTensor
        self.get_pages_func(req_id, num_needed, out_np)
        
        return out_np
        
    def free(self, req_id):
        self.free_func(req_id)
        
    def reset(self):
        self.reset_func()

# Test
if __name__ == "__main__":
    pm = PageManager(10)
    
    print("Allocating 3 pages for Req 0...")
    pages0 = pm.get_pages(0, 3)
    print(f"Req 0 Pages: {pages0}")
    
    print("Allocating 2 pages for Req 1...")
    pages1 = pm.get_pages(1, 2)
    print(f"Req 1 Pages: {pages1}")
    
    print("Appending 2 pages for Req 0 (Total 5)...")
    pages0_new = pm.get_pages(0, 5)
    print(f"Req 0 Pages (Updated): {pages0_new}")
    
    print("Freeing Req 1...")
    pm.free(1)
    
    print("Allocating 4 pages for Req 2 (Should reuse freed pages)...")
    pages2 = pm.get_pages(2, 4)
    print(f"Req 2 Pages: {pages2}")
    
    print("Test Passed!")
