#include "hami/helper/macro.h"
#include "hami/core/task_keys.hpp"
#include "hami/core/reflect.h"

#include "tensorrt_torch/TensorrtInferTensor.hpp"
#include "tensorrt_torch/tensorrt_helper.hpp"

namespace torchpipe {

std::string get_dependency(
    const std::unordered_map<std::string, std::string>& config,
    hami::Backend* this_ptr, const std::string& default_cls_name,
    const std::string& default_dep_name) {
    std::string cls = default_cls_name;
    auto name = HAMI_OBJECT_NAME(hami::Backend, this_ptr);
    if (name) cls = *name;
    auto iter = config.find(cls + "::dependency");
    if (iter != config.end()) {
        cls = iter->second;
    } else {
        cls = default_dep_name;
    }
    return cls;
}

void TensorrtInferTensor::init(
    const std::unordered_map<std::string, std::string>& inconfig,
    const hami::dict& dict_config) {
    auto config = inconfig;
    // handle instance index
    hami::str::try_update(config, TASK_INDEX_KEY, instance_index_);
    hami::str::try_update(config, TASK_INDEX_KEY, instance_num_);
    HAMI_ASSERT(instance_num_ >= 1 && instance_index_ >= 0);

    // initialize converter, get std::shared_ptr<ICudaEngine>
    auto iter = dict_config->find(TASK_ENGINE_KEY);
    HAMI_ASSERT(iter != dict_config->end() &&
                iter.type() == typeid(std::shared_ptr<nvinfer1::ICudaEngine>));

    engine_ = any_cast<std::shared_ptr<nvinfer1::ICudaEngine>>(iter);
    context_ = create_context(engine_.get(), instance_index_);
    info_ = get_context_shape(context_.get(), profile_index);
    HAMI_ASSERT(is_all_positive(info_), "input shape is not positive");
}

void TensorrtInferTensor::forward(const std::vector<hami::dict>& input_output) {
}

// void x::init(const std::unordered_map<std::string, std::string>& config,
//              const hami::dict& dict_config) {
//     // handle instance index
//     hami::str::try_update(config, TASK_INDEX_KEY, instance_index_);
//     hami::str::try_update(config, "instance_num", instance_num_);
//     HAMI_ASSERT(instance_num_ >= 1 && instance_index_ >= 0);

//     // initialize converter, get std::shared_ptr<ICudaEngine>
//     {
//         HAMI_ASSERT(dict_config);
//         auto iter = dict_config->find(TASK_ENGINE_KEY);
//         if (iter != dict_config->end() &&
//             iter->second.type() == typeid(std::string)) {
//             throw std::invalid_argument(
//                 "`" + TASK_ENGINE_KEY +
//                 "` is type of string, not supported yet");
//             return;
//         }
//         {
//             // create engine by converter
//             std::string dependency_name = get_dependency(
//                 config, this, "TensorrtInferTensor", "Onnx2Tensorrt");
//             SPDLOG_INFO("TensorrtInferTensor::dependency={}",
//             dependency_name);

//             auto backend =
//                 std::unique_ptr<Backend>(HAMI_CREATE(Backend,
//                 dependency_name));
//             HAMI_ASSERT(backend,
//                         "`" + dependency_name + "` is not a valid backend");

//             backend->init(config, dict_config);
//         }
//         iter = dict_config->find(TASK_ENGINE_KEY);
//         HAMI_ASSERT(iter != dict_config->end(),
//                     std::string("`") + TASK_ENGINE_KEY +
//                         "` is not found in dict_config");
//         engine_ = hami::any_cast<std::shared_ptr<nvinfer1::ICudaEngine>>(
//             iter->second);
//         HAMI_ASSERT(engine_);
//         if (instance_num_ - 1 == instance_index_) dict_config->erase(iter);
//     }
// }
}  // namespace torchpipe
