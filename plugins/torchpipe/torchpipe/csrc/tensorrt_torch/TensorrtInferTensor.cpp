#include "hami/helper/macro.h"
#include "hami/core/task_keys.hpp"
#include "hami/core/reflect.h"
#include <c10/cuda/CUDAStream.h>

#include "tensorrt_torch/TensorrtInferTensor.hpp"
#include "tensorrt_torch/tensorrt_helper.hpp"
#include <torch/torch.h>
#include "helper/torch.hpp"

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
    hami::str::try_update(config, "instance_num", instance_num_);
    HAMI_ASSERT(instance_num_ >= 1 && instance_index_ >= 0);

    // initialize converter, get std::shared_ptr<ICudaEngine>
    auto iter = dict_config->find(TASK_ENGINE_KEY);
    HAMI_ASSERT(iter != dict_config->end() &&
                iter->second.type() ==
                    typeid(std::shared_ptr<nvinfer1::ICudaEngine>));

    engine_ =
        hami::any_cast<std::shared_ptr<nvinfer1::ICudaEngine>>(iter->second);
    context_ = create_context(engine_.get(), instance_index_);
    info_ = get_context_shape(context_.get(), instance_index_);
    HAMI_ASSERT(is_all_positive(info_), "input shape is not positive");

    (*dict_config)[TASK_IO_INFO_KEY] = std::make_shared<NetIOInfos>(info_);

    // should_change_shape_ =
    //     std::vector<bool>(instaninfo_.first.size() ce_num_, true);
    // binding_ =
    //     std::vector<void*>(info_.first.size() + info_.second.size(),
    //     nullptr);
    if (mem_size_ == 0) mem_size_ = context_->updateDeviceMemorySizeForShapes();

    HAMI_ASSERT(cudaSuccess == cudaEventCreateWithFlags(&input_finish_event_,
                                                        cudaEventDefault));
}

void TensorrtInferTensor::forward(const std::vector<hami::dict>& input_output) {
    HAMI_ASSERT(input_output.size() == 1,
                "only support one (batched) input with explicit batch");

    // input
    const auto inputs =
        hami::dict_gets<torch::Tensor>(input_output[0], TASK_DATA_KEY);

    check_batched_inputs(inputs, info_.first);

    for (unsigned j = 0; j < info_.first.size(); j++) {
        const auto& name_str = *info_.first[j].name;
        const auto* name = name_str.c_str();

        nvinfer1::Dims infer_dims = context_->getTensorShape(name);
        static_assert(sizeof(nvinfer1::Dims) == sizeof(NetIOInfo::Dims64));
        if (!match((NetIOInfo::Dims64*)(&infer_dims), inputs[j])) {
            // should_change_shape_[j] = true;
            context_->setInputShape(name, infer_dims);
            mem_size_ = 0;
        }

        bool status = context_->setTensorAddress(name, inputs[j].data_ptr());
        HAMI_ASSERT(status);
    }

    // output
    auto outputs = hami::dict_gets<torch::Tensor>(input_output[0], "outputs");

    size_t predefined_size = outputs.size();

    for (unsigned j = 0; j < info_.second.size(); j++) {
        const auto& name_str = *info_.first[j].name;
        const auto* name = name_str.c_str();
        const auto infer_dims = context_->getTensorShape(name);

        if (predefined_size > j) {
            HAMI_ASSERT(outputs[j].is_contiguous());
            int64_t total_bytes =
                outputs[j].numel() * outputs[j].element_size();
            int64_t need_bytes =
                std::accumulate(infer_dims.d, infer_dims.d + infer_dims.nbDims,
                                1, std::multiplies<int64_t>()) *
                elementSize(info_.second[j].type);
            if (need_bytes != total_bytes) {
                SPDLOG_ERROR("need_bytes({}) != total_bytes({})", need_bytes,
                             total_bytes);
                HAMI_ASSERT(need_bytes == total_bytes);
            }

        } else {
            outputs.emplace_back(torch::empty(
                std::vector<int64_t>(infer_dims.d,
                                     infer_dims.d + infer_dims.nbDims),
                get_tensor_option(netinfo2torch_type(info_.second[j].type)),
                torch::MemoryFormat::Contiguous));
        }

        HAMI_ASSERT(
            context_->setTensorAddress(name, outputs.back().data_ptr()));
    }

    // memory && execute

#if TRT_USER_MANAGED_MEM
#if NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR <= 7

    // context_->getEngine().getDeviceMemorySizeForProfile(instance_index_);
    if (mem_size_ == 0) mem_size_ = context_->updateDeviceMemorySizeForShapes();

#else

    // context_->getEngine().getDeviceMemorySizeForProfileV2(instance_index_);
    if (mem_size_ == 0) mem_size_ = context_->updateDeviceMemorySizeForShapes();
#endif

    HAMI_ASSERT(mem_size_ > 0, "mem_size is 0");
    const double mem_size_mb = static_cast<double>(mem_size_) / (1024 * 1024);
    SPDLOG_DEBUG("mem_size: {} MB", mem_size_mb);
    torch::Tensor mem = torch_allocate(mem_size_);
#if NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR <= 1
    context_->setDeviceMemory(mem.data_ptr());
#else
    context_->setDeviceMemoryV2(mem.data_ptr(), mem_size_);
#endif
#endif

    HAMI_ASSERT(context_->setInputConsumedEvent(input_finish_event_));

    HAMI_ASSERT(context_->enqueueV3(c10::cuda::getCurrentCUDAStream()));
    cudaEventSynchronize(input_finish_event_);
    inputs_.clear();
    input_output[0]->erase(TASK_DATA_KEY);
    (*input_output[0])[TASK_RESULT_KEY] = outputs;
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

TensorrtInferTensor::~TensorrtInferTensor() {
    cudaEventDestroy(input_finish_event_);
}
}  // namespace torchpipe
