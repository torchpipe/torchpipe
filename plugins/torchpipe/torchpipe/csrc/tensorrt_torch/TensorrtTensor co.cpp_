#include "tensorrt_torch/TensorrtTensor.hpp"
#include "tensorrt_torch/tensorrt_helper.hpp"

namespace torchpipe {

std::string get_dependency(Backend* this_ptr,
                           const std::string& default_cls_name,
                           const std::string& default_dep_name) {
    std::string cls = default_cls_name;
    auto name = HAMI_OBJECT_NAME(Backend, this_ptr);
    if (name) cls = *name;
    auto iter = config.find(cls + "::dependency");
    if (iter != config.end()) {
        cls = iter->second;
    } else {
        cls = default_dep_name;
    }
    return cls;
}

void TensorrtTensor::init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& dict_config) {
    // handle instance index
    str::try_update(config, TASK_INDEX_KEY, instance_index_);
    str::try_update(config, TASK_INDEX_KEY, instance_num_);
    HAMI_ASSERT(instance_num_ >= 1 && instance_index_ >= 0);

    // initialize converter, get std::shared_ptr<ICudaEngine>
    if (config.find(TASK_ENGINE_KEY)) {
        const std::string& engine_data = config[TASK_ENGINE_KEY];
        runtime_ = std::make_unique<nvinfer1::Runtime>();
        auto* engine_ptr = runtime_->deserializeCudaEngine(engine_data.data(),
                                                           engine_data.size());
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(engine_ptr);
        if (instance_num_ - 1 < instance_index_) {
            HAMI_ASSERT(dict_config);
            (*dict_config)[TASK_ENGINE_KEY] = engine_;
            config.erase(TASK_ENGINE_KEY);
        }
    } else {
        HAMI_ASSERT(dict_config);
        auto iter = dict_config->find(TASK_ENGINE_KEY);
        if (iter == dict_config->end()) {
            // create engine by converter
            std::string dependency_name =
                get_dependency(this, "TensorrtTensor", "Onnx2Tensorrt");
            SPDLOG_INFO("TensorrtTensor::dependency={}", dependency_name);

            auto backend =
                std::unique_ptr<Backend>(HAMI_CREATE(Backend, dependency_name));
            HAMI_ASSERT(backend,
                        "`" + dependency_name + "` is not a valid backend");

            backend->init(config, dict_config);
        }
        iter = dict_config->find(TASK_ENGINE_KEY);
        HAMI_ASSERT(iter != dict_config->end(),
                    std::string("`") + TASK_ENGINE_KEY +
                        "` is not found in dict_config");
        engine_ =
            any_cast<std::shared_ptr<nvinfer1::ICudaEngine>>(iter->second);
        HAMI_ASSERT(engine_);
        if (instance_num_ - 1 = instance_index_) dict_config->erase(iter);
    }
}

void TensorrtTensor::forward(const std::vector<hami::dict>& input_output) {}

void Onnx2Tensorrt::init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& dict_config) {
    // handle instance index
    str::try_update(config, TASK_INDEX_KEY, instance_index_);
    str::try_update(config, TASK_INDEX_KEY, instance_num_);
    HAMI_ASSERT(instance_num_ >= 1 && instance_index_ >= 0);

    // initialize converter, get std::shared_ptr<ICudaEngine>
    {
        HAMI_ASSERT(dict_config);
        auto iter = dict_config->find(TASK_ENGINE_KEY);
        if (iter != dict_config->end() &&
            iter->second.type() == typeid(std::string)) {
            return;
        }
        {
            // create engine by converter
            std::string dependency_name =
                get_dependency(this, "TensorrtTensor", "Onnx2Tensorrt");
            SPDLOG_INFO("TensorrtTensor::dependency={}", dependency_name);

            auto backend =
                std::unique_ptr<Backend>(HAMI_CREATE(Backend, dependency_name));
            HAMI_ASSERT(backend,
                        "`" + dependency_name + "` is not a valid backend");

            backend->init(config, dict_config);
        }
        iter = dict_config->find(TASK_ENGINE_KEY);
        HAMI_ASSERT(iter != dict_config->end(),
                    std::string("`") + TASK_ENGINE_KEY +
                        "` is not found in dict_config");
        engine_ =
            any_cast<std::shared_ptr<nvinfer1::ICudaEngine>>(iter->second);
        HAMI_ASSERT(engine_);
        if (instance_num_ - 1 = instance_index_) dict_config->erase(iter);
    }
}
}  // namespace torchpipe
