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
  if (iter != config.end()) {
    independent_thread_index_ = std::stoi(iter->second);
    HAMI_ASSERT(independent_thread_index_ >= 0);
  } else {
    SPDLOG_WARN(
        "You are using an independent CUDA stream, "
        "but not in an independent thread mode(TASK_INDEX_KEY was not "
        "detected). ");
    HAMI_ASSERT(
        dep,
        "You are not in an independent thread mode, pls set the dependency by " +
            cls_name + "[*]");
  }

  if (config.find("device_id") != config.end()) {
    throw std::runtime_error(
        "SyncTensor: device_id is not supported by SyncTensor yet.");
  }

  const auto device_id_int = -1;

  c10::cuda::getCurrentCUDAStream().synchronize();

  if (independent_thread_index_ >= 0) {
    bNeedSync_ = torch_not_use_default_stream(device_id_int, true);
    // Schedule保证了init和forward在同一个线程
    SPDLOG_INFO("SyncTensor: sync enabled={}", bNeedSync_);
  } else if (
      c10::cuda::getCurrentCUDAStream(device_id_int) ==
      c10::cuda::getDefaultCUDAStream(device_id_int)) {
    // bNeedSync_ = true;
    stream_ = c10::cuda::getStreamFromPool(true, device_id_int);

    [[maybe_unused]] c10::cuda::CUDAStreamGuard s(*stream_);
    HAMI_ASSERT(
        c10::cuda::getCurrentCUDAStream(-1) !=
        c10::cuda::getDefaultCUDAStream(-1));
    HAMI_ASSERT(dep);

    owned_backend_ = init_backend(*dep, config, kwargs);
    c10::cuda::getCurrentCUDAStream().synchronize();
  }
  if (dep && !owned_backend_) {
    owned_backend_ = init_backend(*dep, config, kwargs);
  }

  c10::cuda::getCurrentCUDAStream().synchronize();

  HAMI_ASSERT(c10::cuda::device_count() >= 1);

  return;
}

// void SyncTensor::post_init(
//     const std::unordered_map<std::string, std::string>& config,
//     const dict& kwargs) {
//   c10::cuda::getCurrentCUDAStream().synchronize();
// }

void SyncTensor::impl_forward(const std::vector<dict>& ios) {
  if (independent_thread_index_ >= 0) {
    impl_dep_forward(ios);
    if (bNeedSync_)
      c10::cuda::getCurrentCUDAStream().synchronize();
  } else if (stream_) {
    [[maybe_unused]] c10::cuda::CUDAStreamGuard s(*stream_);
    assert(
        c10::cuda::getCurrentCUDAStream(-1) !=
        c10::cuda::getDefaultCUDAStream(-1));
    assert(stream_ == c10::cuda::getCurrentCUDAStream(-1));

    event_.record(s.original_stream());
    event_.block(s.current_stream());

    owned_backend_->forward(ios);
    s.current_stream().synchronize();
  } else {
    impl_dep_forward(ios);
  }
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

HAMI_REGISTER_BACKEND(TorchStreamPool);

} // namespace torchpipe