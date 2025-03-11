#include "c10/cuda/CUDAStream.h"
#include "c10/cuda/CUDAFunctions.h"

#include "hami/core/helper.hpp"
#include "torchplugins/SyncTensor.hpp"
#include "helper/torch.hpp"

namespace torchpipe {
void SyncTensor::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& dict_config) {
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

    assert(independent_thread_index >= 0);

    auto device_id_int = -1;  // std::stoi(device_id);
    bool high_priority = config.find("high_priority") != config.end();
    high_priority = high_priority && (independent_thread_index < 32);

    bNeedSync_ = torch_not_use_default_stream(device_id_int, high_priority);
    SPDLOG_DEBUG("SyncTensor: sync enabled={} high_priority={}", bNeedSync_,
                 high_priority);

    c10::InferenceMode guard;  // optinal

    // c10::cuda::getCurrentCUDAStream().synchronize();

    HAMI_ASSERT(c10::cuda::device_count() >= 1);

    return;
}

void SyncTensor::post_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& dict_config) {
    c10::cuda::getCurrentCUDAStream().synchronize();
}

void SyncTensor::custom_forward_with_dep(const std::vector<dict>& input_dicts,
                                         Backend* dependency) {
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

HAMI_REGISTER(hami::Backend, SyncTensor,
              "SyncTensor,TorchStreamGuard,CUDAStreamGuard,StreamGuard");
}  // namespace torchpipe