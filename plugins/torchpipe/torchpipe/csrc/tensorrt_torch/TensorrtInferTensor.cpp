#include "hami/helper/macro.h"
#include "hami/helper/timer.hpp"

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
    hami::Backend* this_ptr,
    const std::string& default_cls_name,
    const std::string& default_dep_name) {
  std::string cls = default_cls_name;
  auto name = HAMI_OBJECT_NAME(hami::Backend, this_ptr);
  if (name)
    cls = *name;
  auto iter = config.find(cls + "::dependency");
  if (iter != config.end()) {
    cls = iter->second;
  } else {
    cls = default_dep_name;
  }
  return cls;
}

void TensorrtInferTensor::impl_init(
    const std::unordered_map<std::string, std::string>& inconfig,
    const hami::dict& kwargs) {
  auto config = inconfig;
  // handle instance index
  hami::str::try_update(config, TASK_INDEX_KEY, instance_index_);
  hami::str::try_update(config, "instance_num", instance_num_);

  HAMI_ASSERT(instance_num_ >= 1 && instance_index_ >= 0);

  if (instance_num_ > 1) {
    HAMI_ASSERT(
        !torch_is_using_default_stream(),
        "In multi-instance mode, the default stream is prohibited. "
        "Please use a dedicated CUDA stream with StreamGuard: "
        "e.g., StreamGuard[X], "
        "or S_v0[X1, X2, StreamGuard]");
  }

  auto iter = kwargs->find(TASK_ENGINE_KEY);
  HAMI_ASSERT(
      iter != kwargs->end() &&
      iter->second.type() == typeid(nvinfer1::ICudaEngine*));

  engine_ = hami::any_cast<nvinfer1::ICudaEngine*>(iter->second);
  context_ = create_context(engine_, instance_index_);
  info_ = get_context_shape(context_.get(), instance_index_);
  HAMI_ASSERT(is_all_positive(info_), "input shape is not positive");

  (*kwargs)[TASK_IO_INFO_KEY] = std::make_shared<NetIOInfos>(info_);

  if (mem_size_ == 0)
    mem_size_ = context_->updateDeviceMemorySizeForShapes();

  HAMI_ASSERT(
      cudaSuccess ==
      cudaEventCreateWithFlags(&input_finish_event_, cudaEventDefault));
}

void TensorrtInferTensor::impl_forward(
    const std::vector<hami::dict>& input_output) {
  HAMI_ASSERT(
      input_output.size() == 1,
      "only support one (batched) input with explicit batch");

  // input
  auto inputs = hami::dict_gets<torch::Tensor>(input_output[0], TASK_DATA_KEY);

  check_batched_inputs(inputs, info_.first);

  // hami::helper::ScopedTimer timer("tensorrt infer. size = " +
  // std::to_string(inputs[0].sizes()[0]), 0.01);

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

  // outputs from user
  std::vector<torch::Tensor> outputs;
  if (input_output[0]->find(hami::TASK_OUTPUT_KEY) != input_output[0]->end())
    outputs =
        hami::dict_gets<torch::Tensor>(input_output[0], hami::TASK_OUTPUT_KEY);

  size_t predefined_size = outputs.size();

  for (unsigned j = 0; j < info_.second.size(); j++) {
    const auto& name_str = *info_.second[j].name;
    const auto* name = name_str.c_str();
    const auto infer_dims = context_->getTensorShape(name);
    HAMI_FATAL_ASSERT(
        infer_dims.nbDims > 0,
        "TensorRT output tensor shape is empty. "
        "Please check the model and the input shape.");

    if (predefined_size > j) {
      HAMI_ASSERT(outputs[j].is_contiguous());
      int64_t total_bytes = outputs[j].numel() * outputs[j].element_size();
      int64_t need_bytes = std::accumulate(
                               infer_dims.d,
                               infer_dims.d + infer_dims.nbDims,
                               1,
                               std::multiplies<int64_t>()) *
          elementSize(info_.second[j].type);
      if (need_bytes != total_bytes) {
        SPDLOG_ERROR(
            "need_bytes({}) != total_bytes({})", need_bytes, total_bytes);
        HAMI_ASSERT(need_bytes == total_bytes);
      }
    } else {
      outputs.emplace_back(torch::empty(
          std::vector<int64_t>(infer_dims.d, infer_dims.d + infer_dims.nbDims),
          get_tensor_option(netinfo2torch_type(info_.second[j].type)),
          torch::MemoryFormat::Contiguous));
    }

    HAMI_ASSERT(context_->setTensorAddress(name, outputs.back().data_ptr()));
  }

  // memory && execute

#if TRT_USER_MANAGED_MEM
#if NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR <= 7

  // context_->getEngine().getDeviceMemorySizeForProfile(instance_index_);
  if (mem_size_ == 0)
    mem_size_ = context_->updateDeviceMemorySizeForShapes();

#else

  if (mem_size_ == 0)
    mem_size_ = context_->updateDeviceMemorySizeForShapes();
  // if (mem_size_ == 0) mem_size_ =
  // [[maybe_unused]] auto zz = context_->updateDeviceMemorySizeForShapes();
  // [[maybe_unused]] auto tt =
  //     context_->getEngine().getDeviceMemorySizeForProfile(instance_index_);
  // [[maybe_unused]] auto tt2 = context_->getEngine().getDeviceMemorySize();

  // bool use_v2 = context_->getEngine().getWeightStreamingBudgetV2() <
  //               context_->getEngine().getStreamableWeightsSize();
#endif
  torch::Tensor mem;
  void* mem_ptr = nullptr;
  if (mem_size_ > 0) {
    // const double mem_size_mb =
    //     static_cast<double>(mem_size_) / (1024 * 1024);
    // SPDLOG_INFO("model context memory size: {} MB", mem_size_mb);
    mem = torch_allocate(mem_size_);
    mem_ptr = mem.data_ptr();
#if NV_TENSORRT_MAJOR < 10 && NV_TENSORRT_MINOR < 1
    context_->setDeviceMemory(mem_ptr);
#else
    // if (use_v2)
    //     context_->setDeviceMemoryV2(mem.data_ptr(), mem_size_);
    // else
    context_->setDeviceMemory(mem_ptr); // todo
#endif
  }
#endif

  HAMI_ASSERT(context_->setInputConsumedEvent(input_finish_event_));

  HAMI_ASSERT(
      context_->enqueueV3(c10::cuda::getCurrentCUDAStream()),
      "TensorRT inference execution failed. Check TensorRT logs for "
      "detailed error information.");
  cudaEventSynchronize(input_finish_event_);
  inputs.clear();
  input_output[0]->erase(TASK_DATA_KEY);
  if (outputs.size() == 1) {
    (*input_output[0])[TASK_RESULT_KEY] = outputs[0];
  } else {
    (*input_output[0])[TASK_RESULT_KEY] = outputs;
  }
}

TensorrtInferTensor::~TensorrtInferTensor() {
  cudaEventDestroy(input_finish_event_);
}

HAMI_REGISTER(hami::Backend, TensorrtInferTensor);
} // namespace torchpipe
