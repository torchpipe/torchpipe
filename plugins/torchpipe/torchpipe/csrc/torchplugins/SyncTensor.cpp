#include "c10/cuda/CUDAFunctions.h"
#include "c10/cuda/CUDAStream.h"

#include "ATen/cuda/CUDAEvent.h"
#include "helper/torch.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/helper/resource_pool.hpp"
#include "torchplugins/SyncTensor.hpp"

#include <tvm/ffi/extra/c_env_api.h>
#include "helper/dlpack_helper.hpp"

namespace torchpipe {
void SyncTensor::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  auto dep = omniback::parser_v2::get_opt_dependency_name(this, config);

  const auto cls_name = omniback::get_cls_name(this, "SyncTensor");

  auto iter = config.find(TASK_INDEX_KEY);
  OMNI_ASSERT(
      iter != config.end(),
      "You are not in an independent thread mode(TASK_INDEX_KEY was not detected. Maybe use `With[StreamPool, *]` instead");

  independent_thread_index_ = std::stoi(iter->second);
  OMNI_ASSERT(independent_thread_index_ >= 0);

  if (config.find("device_id") != config.end()) {
    throw std::runtime_error(
        "SyncTensor: device_id is not supported by SyncTensor yet.");
  }

  const auto device_id_int = c10::cuda::current_device() ;//- 1;

  c10::cuda::getCurrentCUDAStream().synchronize();

   
  
  bNeedSync_ =
      torch_not_use_default_stream(device_id_int, true);
  // Schedule保证了init和forward在同一个线程
  OMNI_ASSERT(
      bNeedSync_,
      "This backend can only be used in default current stream. may be use `With[StreamPool,*]` instead.");

  TVMFFIStreamHandle out_original_stream{nullptr};
  TVMFFIStreamHandle in_stream = c10::cuda::getCurrentCUDAStream().stream();
  TVM_FFI_ICHECK(
      0 ==
      TVMFFIEnvSetStream(
          kDLCUDA, device_id_int, in_stream, &out_original_stream));
  DLPackManagedTensorAllocator opt_out_original_allocator{nullptr};
  TVM_FFI_ICHECK(nullptr == TVMFFIEnvGetDLPackManagedTensorAllocator());
      // https: //
      // github.com/apache/tvm-ffi/blob/6e7cafab78cb007d066bc860c600e2ba80b4d1a7/python/tvm_ffi/utils/_build_optional_torch_c_dlpack.py#L535
  TVM_FFI_ICHECK(
          0 ==
          TVMFFIEnvSetDLPackManagedTensorAllocator(
              torch_allocator(), 0, &opt_out_original_allocator));
  TVM_FFI_ICHECK(nullptr == out_original_stream);
  // TVM_FFI_ICHECK(nullptr == opt_out_original_allocator);

  if (dep && !owned_backend_) {
    owned_backend_ = omniback::init_backend(*dep, config, kwargs);
  }
  // ManagedTensorAllocator

  c10::cuda::getCurrentCUDAStream().synchronize();

  OMNI_ASSERT(c10::cuda::device_count() >= 1);

  return;
}

void SyncTensor::impl_forward(const std::vector<dict>& ios) {
  // std::string sync_stream = dict_get<std::string>(ios[0],
  // "sync_stream",true);

  if (owned_backend_) {
    static auto curr_stream = c10::cuda::getCurrentCUDAStream(-1);
    static auto default_stream = c10::cuda::getDefaultCUDAStream(-1);

    event_.record(default_stream);
    event_.block(curr_stream);
  }
  impl_dep_forward(ios);

  c10::cuda::getCurrentCUDAStream().synchronize();
}

OMNI_REGISTER(omniback::Backend, SyncTensor, "SyncTensor,StreamGuard");

class TorchStreamPool : public omniback::Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {
    auto [args, kwargs] =
        omniback::parser_v2::get_args_kwargs(this, "TorchStreamPool", params);
    omniback::str::try_update<size_t>(kwargs, "max_stream", max_stream_count_);
    OMNI_ASSERT(max_stream_count_ > 0 && max_stream_count_ < 32);
    stream_pool_ = std::make_unique<omniback::pool::ResourcePool<size_t>>(
        max_stream_count_);
    for (size_t i = 0; i < max_stream_count_; ++i) {
      auto stream = c10::cuda::getStreamFromPool(true, -1);
      stream_event_.emplace_back(
          StreamWithEvent{std::move(stream), at::cuda::CUDAEvent()});
    }
  }

  void impl_forward_with_dep(
      const std::vector<omniback::dict>& ios,
      omniback::Backend& dep) override {
    auto index = stream_pool_->acquire();
    omniback::pool::ResourcePool<size_t>::lease_guard guard(
        stream_pool_.get(), index);
    auto& se = stream_event_.at(index);
    auto original_stream = c10::cuda::getCurrentCUDAStream(-1);
    if (se.stream != original_stream) {
      c10::cuda::CUDAStreamGuard s(se.stream);
      // https://stackoverflow.com/questions/15501699/cudastreamwaitevent-does-not-seem-to-wait
      se.event.record(original_stream);
      se.event.block(se.stream);

      dep.safe_forward(ios);

      se.event.record(se.stream);
      se.event.block(original_stream);
    } else {
      dep.safe_forward(ios);
    }
  }

  [[nodiscard]] uint32_t impl_max() const override {
    return std::numeric_limits<uint32_t>::max();
  }

 private:
  struct StreamWithEvent {
    c10::cuda::CUDAStream stream;
    at::cuda::CUDAEvent event;
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446
  };
  size_t max_stream_count_{1};
  std::vector<StreamWithEvent> stream_event_;
  std::unique_ptr<omniback::pool::ResourcePool<size_t>> stream_pool_;
};

OMNI_REGISTER(
    omniback::Backend,
    TorchStreamPool,
    "TorchStreamPool, StreamPool");
} // namespace torchpipe
