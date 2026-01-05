#pragma once
#include <string>
#include <unordered_set>

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include "omniback/ffi/types.hpp"


namespace torchpipe {

inline DLPackManagedTensorAllocator torch_allocator() {
  DLPackExchangeAPI* api =
      reinterpret_cast<DLPackExchangeAPI*>(omniback::ffi::dlpack_exchange_api());
  if (api)
   { 
    static DLPackManagedTensorAllocator& alloc = api->managed_tensor_allocator;
    TVM_FFI_ICHECK(alloc);
    return alloc;
  }else{
    return nullptr;
  }
}

}