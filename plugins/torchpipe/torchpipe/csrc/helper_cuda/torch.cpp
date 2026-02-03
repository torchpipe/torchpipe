// Copyright 2021-2026 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#if 1
#include "c10/cuda/CUDAStream.h"
#endif
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

// #include "time_utils.hpp"
// #include "base_logging.hpp"
// #include <torch/serialize.h>
// #include <torch/extension.h>
#include <torch/torch.h>
#include <fstream>

#include <omniback/extension.hpp>
#include "helper_cuda/torch.hpp"
// #include "NvInferRuntime.h"
#include "omniback/helper/timer.hpp"
#include "helper/dlpack_helper.hpp"
#include <tvm/ffi/container/tensor.h>
#include "omniback/addons/torch/type_traits.h"

namespace torchpipe {
namespace{

inline c10::cuda::CUDAStream get_current_stream() {
  return c10::cuda::getCurrentCUDAStream();
}

// GPU事件初始化（线程安全版）
inline const at::cuda::CUDAEvent& start_event() {
  static at::cuda::CUDAEvent ev;
  static std::once_flag flag;
  std::call_once(flag, [&] { ev.record(at::cuda::getDefaultCUDAStream()); });
  return ev;
}
}

bool torch_not_use_default_stream(bool high_prio) {
  if (c10::cuda::getCurrentCUDAStream() == c10::cuda::getDefaultCUDAStream()) {
    c10::cuda::setCurrentCUDAStream(
        c10::cuda::getStreamFromPool(
            high_prio,
            -1)); // Schedule保证了init和forward在同一个线程
    return true;
  }
  return false;
}

bool torch_not_use_default_stream(int device_id, bool high_prio) {
  // if (c10::cuda::current_device() != device_id && device_id >= 0) {
  //   c10::cuda::set_device(device_id);
  // }
  OMNI_ASSERT(device_id < 0 || c10::cuda::current_device() == device_id);
  if (c10::cuda::getCurrentCUDAStream(device_id) ==
      c10::cuda::getDefaultCUDAStream(device_id)) {
    c10::cuda::setCurrentCUDAStream(
        c10::cuda::getStreamFromPool(
            high_prio, device_id)); // Schedule保证了init和forward在同一个线程
    return true;
  }
  return false;
}

bool torch_is_using_default_stream() {
  if (c10::cuda::getCurrentCUDAStream(-1) ==
      c10::cuda::getDefaultCUDAStream(-1)) {
    return true;
  }
  return false;
}

// https://discuss.pytorch.org/t/asynchronous-copy-in-c-when-input-has-been-destructed/186515
torch::Tensor to_current_device(torch::Tensor input) {
  if (input.device() == torch::kCPU)
    return input.cuda();
  if (input.device().index() == c10::cuda::current_device())
    return input;
  torch::TensorOptions options;
  // input.is_pinned()
  return input.to(
      torch::TensorOptions().device(torch::kCUDA, -1),
      false,
      false,
      input.suggest_memory_format()); // 这里为异步操作, pytorch 自身cache
                                      // pinned memory， 不怕析构
} 

void copy2ptr(torch::Tensor input, char* ptr) {
  OMNI_ASSERT(
      input.is_contiguous(), "copy2ptr: input tensor must be contiguous");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  size_t size = input.numel() * input.element_size();
  OMNI_ASSERT(
      cudaMemcpyAsync(
          ptr, input.data_ptr(), size, cudaMemcpyDeviceToDevice, stream) ==
          cudaSuccess,
      "copy2ptr: cudaMemcpyAsync failed");
}



// 获取当前CUDA流的时间（毫秒），对齐CPU时间
float cuda_time() {
  // 记录GPU结束事件
  at::cuda::CUDAEvent stop_event;
  stop_event.record(get_current_stream());
  stop_event.synchronize();

  // 计算GPU时间
  float gpu_ms = start_event().elapsed_time(stop_event);

  // 初始化时间偏移量（只执行一次）
  static float time_offset = [&]() {
    start_event().synchronize();
    at::cuda::CUDAEvent sync_event;
    sync_event.record(get_current_stream());
    sync_event.synchronize();

    auto cpu_elapsed = om::helper::timestamp();
    return cpu_elapsed - start_event().elapsed_time(sync_event);
  }();

  // 返回对齐后的时间
  return gpu_ms + time_offset;
}


int torch_malloc_async(
    void* ctx,
    void** p,
    size_t size,
    cudaStream_t stream) {
  (void)ctx; // Ignore the context pointer if not used
  if (size == 0) {
    *p = nullptr;
    return -1; // Error: Invalid size
  }

  *p = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, stream);
  if (*p == nullptr) {
    return -2; // Error: Memory allocation failed
  }

  return 0; // Success
}

// Async Memory Free with Error Handling
int torch_free_async(
    void* ctx,
    void* p,
    size_t size,
    cudaStream_t stream) {
  (void)ctx; // Ignore the context pointer if not used
  if (p == nullptr) {
    return -3; // Error: Invalid pointer
  }

  c10::cuda::CUDACachingAllocator::raw_delete(p);
  return 0; // Success
}

} // namespace torchpipe