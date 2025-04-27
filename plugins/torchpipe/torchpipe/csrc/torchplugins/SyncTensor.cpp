#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAFunctions.h"

#include "hami/core/helper.hpp"
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
    HAMI_ASSERT(
        c10::cuda::getCurrentCUDAStream(-1) !=
        c10::cuda::getDefaultCUDAStream(-1));

    event_.record(s.original_stream());
    event_.block(s.current_stream());

    owned_backend_->forward(ios);
    c10::cuda::getCurrentCUDAStream().synchronize();
  } else {
    impl_dep_forward(ios);
  }
}

HAMI_REGISTER(hami::Backend, SyncTensor, "SyncTensor,StreamGuard");

class TorchStreamPool : public hami::Backend {
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override {}
  void impl_forward_with_dep(
      const std::vector<hami::dict>& io,
      hami::Backend* dependency) override {}

  [[nodiscard]] size_t impl_max() const override {
    return std::numeric_limits<size_t>::max();
  }
};

HAMI_REGISTER_BACKEND(TorchStreamPool);

} // namespace torchpipe