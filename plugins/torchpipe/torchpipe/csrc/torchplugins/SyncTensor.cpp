#include "c10/cuda/CUDAStream.h"

#include "torchplugins/SyncTensor.hpp"
#include "helper/torch.hpp"

namespace torchpipe {
void SyncTensor::pre_init(const std::unordered_map<std::string, std::string>& config,
                          const dict& dict_config) {
  set_dependency_name("Identity");

  auto iter = config.find(TASK_INDEX_KEY);
  if (iter != config.end()) {
    auto device_id_int = -1;  // std::stoi(device_id);
    int independent_thread_index = std::stoi(iter->second);
    assert(independent_thread_index >= 0);
    bool high_priority = config.find("high_priority") != config.end();
    high_priority = high_priority && (independent_thread_index < 32);

    bNeedSync_ = torch_not_use_default_stream(device_id_int, high_priority);
    SPDLOG_DEBUG("SyncTensor: sync enabled={} high_priority={}", bNeedSync_, high_priority);
  }

  if (config.find("device_id") != config.end()) {
    throw std::runtime_error("SyncTensor: device_id is not supported by SyncTensor yet.");
  }

  c10::InferenceMode guard;  // optinal

  return;
}

void SyncTensor::post_init(const std::unordered_map<std::string, std::string>& config,
                           const dict& dict_config) {
  c10::cuda::getCurrentCUDAStream().synchronize();
}

void SyncTensor::forward_impl(const std::vector<dict>& input_dicts, Backend* dependency) {
  c10::InferenceMode guard;  // optinal

  if (bNeedSync_) {
    try {
      dependency->forward(input_dicts);
    } catch (...) {
      c10::cuda::getCurrentCUDAStream().synchronize();
      std::rethrow_exception(std::current_exception());
    }
    c10::cuda::getCurrentCUDAStream().synchronize();
  } else {
    dependency->forward(input_dicts);
  }
}

HAMI_REGISTER(hami::Backend, SyncTensor, "SyncTensor");
}  // namespace torchpipe