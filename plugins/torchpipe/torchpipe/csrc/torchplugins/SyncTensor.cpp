#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAFunctions.h"

#include "hami/core/helper.hpp"
#include "torchplugins/SyncTensor.hpp"
#include "helper/torch.hpp"

namespace torchpipe {
void SyncTensor::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  set_dependency_name(config, "SyncTensor", "Identity");
  const auto cls_name = hami::get_cls_name(this, "SyncTensor");

  auto iter = config.find(TASK_INDEX_KEY);
  int independent_thread_index = 0;
  if (iter != config.end()) {
    independent_thread_index = std::stoi(iter->second);
  } else {
    SPDLOG_WARN(
        "You are using an independent CUDA stream, "
        "but not in an independent thread mode(TASK_INDEX_KEY was not "
        "detected). "
        " If you want to run in the current "
        "stream, remove this {}. see "
        "https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams",
        cls_name);
  }

  if (config.find("device_id") != config.end()) {
    throw std::runtime_error(
        "SyncTensor: device_id is not supported by SyncTensor yet.");
  }

  HAMI_ASSERT(independent_thread_index >= 0);

  auto device_id_int = -1; // std::stoi(device_id);
  // bool high_priority = config.find("high_priority") != config.end();
  // high_priority = high_priority && (independent_thread_index < 32);

  bNeedSync_ = torch_not_use_default_stream(device_id_int, true);
  SPDLOG_INFO("SyncTensor: sync enabled={}", bNeedSync_);

  c10::InferenceMode guard; // optinal

  // c10::cuda::getCurrentCUDAStream().synchronize();

  HAMI_ASSERT(c10::cuda::device_count() >= 1);

  return;
}

void SyncTensor::post_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  c10::cuda::getCurrentCUDAStream().synchronize();
}

void SyncTensor::custom_forward_with_dep(
    const std::vector<dict>& input_dicts,
    Backend* dependency) {
  c10::InferenceMode guard; // optinal

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

HAMI_REGISTER(hami::Backend, SyncTensor, "SyncTensor,ThreadStreamGuard");

void StreamGuard::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  set_dependency_name(
      config, "StreamGuard", "UserMustProvideThisBackendNameForStreamGuard[*]");
  const auto cls_name = hami::get_cls_name(this, "StreamGuard");

  if (config.find("device_id") != config.end()) {
    throw std::runtime_error(
        "StreamGuard: device_id is not supported by StreamGuard yet.");
  }

  auto device_id_int = -1; // std::stoi(device_id);
  // bool high_priority = config.find("high_priority") != config.end();
  // high_priority = high_priority && (independent_thread_index < 32);

  if (c10::cuda::getCurrentCUDAStream(device_id_int) ==
      c10::cuda::getDefaultCUDAStream(device_id_int)) {
    stream_ = c10::cuda::getStreamFromPool(
        true, device_id_int); // Schedule保证了init和forward在同一个线程
    stream_guard_ = std::make_unique<c10::cuda::CUDAStreamGuard>(*stream_);
  }

  // c10::cuda::getStreamFromExternal(stream, device_id)

  SPDLOG_INFO("StreamGuard: sync enabled={}", bool(stream_));

  [[maybe_unused]] c10::InferenceMode guard; // optinal

  // c10::cuda::getCurrentCUDAStream().synchronize();

  HAMI_ASSERT(c10::cuda::device_count() >= 1);

  return;
}

void StreamGuard::post_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  if (stream_guard_)
    try {
      c10::cuda::getCurrentCUDAStream().synchronize();
    } catch (...) {
      stream_guard_.release();
      std::rethrow_exception(std::current_exception());
    }
}

void StreamGuard::custom_forward_with_dep(
    const std::vector<dict>& input_dicts,
    Backend* dependency) {
  [[maybe_unused]] c10::InferenceMode guard; // optinal

  if (stream_) {
    const auto custom_delete = [this](void*) {
      try {
        c10::cuda::getCurrentCUDAStream().synchronize();
      } catch (...) {
        // stream_guard_.release();
        std::rethrow_exception(std::current_exception());
      }
      // stream_guard_.release();
    };
    std::unique_ptr<void, std::function<void(void*)>> stream_guard(
        nullptr, custom_delete);
    // stream_guard_ = std::make_unique<c10::cuda::CUDAStreamGuard>(*stream_);
    [[maybe_unused]] c10::cuda::CUDAStreamGuard s(*stream_);
    // at::cuda::CUDAEvent caller_exec_complete;
    // caller_exec_complete.record(compiled_engine->caller_stream);
    // caller_exec_complete.block(compiled_engine->engine_stream);
    // SPDLOG_INFO(
    //     "xxxx {}, {} {}",
    //     (long long)(*stream_).stream(),
    //     (long long)c10::cuda::getCurrentCUDAStream(-1).stream(),
    //     (long long)c10::cuda::getDefaultCUDAStream(-1).stream());
    // HAMI_ASSERT(*stream_ == c10::cuda::getCurrentCUDAStream(-1));
    HAMI_ASSERT(
        c10::cuda::getCurrentCUDAStream(-1) !=
        c10::cuda::getDefaultCUDAStream(-1));
    dependency->forward(input_dicts);

  } else {
    dependency->forward(input_dicts);
  }
}

HAMI_REGISTER(hami::Backend, StreamGuard, "TorchStreamGuard,StreamGuard");
} // namespace torchpipe