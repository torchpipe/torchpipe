#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAFunctions.h"

#include "hami/core/helper.hpp"
#include "hami/helper/resource_pool.hpp"
#include "torchplugins/SyncTensor.hpp"
#include "helper/torch.hpp"
#include "ATen/cuda/CUDAEvent.h"

namespace torchpipe {
void SyncTensor::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  auto dep = hami::parser_v2::get_opt_dependency_name(this, config);

  const auto cls_name = hami::get_cls_name(this, "SyncTensor");

  auto iter = config.find(TASK_INDEX_KEY);
  HAMI_ASSERT(
      iter != config.end(),
      "You are not in an independent thread mode(TASK_INDEX_KEY was not detected. Maybe use `With[StreamPool, *]` instead");

  independent_thread_index_ = std::stoi(iter->second);
  HAMI_ASSERT(independent_thread_index_ >= 0);

  if (config.find("device_id") != config.end()) {
    throw std::runtime_error(
        "SyncTensor: device_id is not supported by SyncTensor yet.");
  }

  const auto device_id_int = -1;

  c10::cuda::getCurrentCUDAStream().synchronize();

  bNeedSync_ = torch_not_use_default_stream(device_id_int, true);
  // Schedule保证了init和forward在同一个线程
  HAMI_ASSERT(
      bNeedSync_,
      "This backend can only be used in default current stream. may be use `With[StreamPool,*]` instead.");

  if (dep && !owned_backend_) {
    owned_backend_ = init_backend(*dep, config, kwargs);
  }

  c10::cuda::getCurrentCUDAStream().synchronize();

  HAMI_ASSERT(c10::cuda::device_count() >= 1);

  return;
}

void SyncTensor::impl_forward(const std::vector<dict>& ios) {
  // std::string sync_stream = dict_get<std::string>(ios[0], "sync_stream",true);
  
  if (owned_backend_) {
    static auto curr_stream = c10::cuda::getCurrentCUDAStream(-1);
    static auto default_stream = c10::cuda::getDefaultCUDAStream(-1);

    event_.record(default_stream);
    event_.block(curr_stream);
  }
  impl_dep_forward(ios);

  c10::cuda::getCurrentCUDAStream().synchronize();
}

HAMI_REGISTER(hami::Backend, SyncTensor, "SyncTensor,StreamGuard");

class TorchStreamPool : public hami::Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override {
    auto [args, kwargs] =
        hami::parser_v2::get_args_kwargs(this, "TorchStreamPool", params);
    hami::str::try_update<size_t>(kwargs, "max_stream", max_stream_count_);
    HAMI_ASSERT(max_stream_count_ > 0 && max_stream_count_ < 32);
    stream_pool_ =
        std::make_unique<hami::pool::ResourcePool<size_t>>(max_stream_count_);
    for (size_t i = 0; i < max_stream_count_; ++i) {
      auto stream = c10::cuda::getStreamFromPool(true, -1);
      stream_event_.emplace_back(
          StreamWithEvent{std::move(stream), at::cuda::CUDAEvent()});
    }
  }

  void impl_forward_with_dep(
      const std::vector<hami::dict>& ios,
      hami::Backend& dep) override {
    auto index = stream_pool_->acquire();
    hami::pool::ResourcePool<size_t>::lease_guard guard(
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

  [[nodiscard]] size_t impl_max() const override {
    return std::numeric_limits<size_t>::max();
  }

 private:
  struct StreamWithEvent {
    c10::cuda::CUDAStream stream;
    at::cuda::CUDAEvent event;
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446
  };
  size_t max_stream_count_{1};
  std::vector<StreamWithEvent> stream_event_;
  std::unique_ptr<hami::pool::ResourcePool<size_t>> stream_pool_;
};

HAMI_REGISTER(hami::Backend, TorchStreamPool, "TorchStreamPool, StreamPool");
} // namespace torchpipe
