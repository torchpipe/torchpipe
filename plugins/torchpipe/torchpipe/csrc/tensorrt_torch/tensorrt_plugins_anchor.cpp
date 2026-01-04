#include <cstdlib>
#include <cstring>
#include <exception>

#include <vector>

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#if NV_TENSORRT_MAJOR >= 10
#include <NvInferRuntimePlugin.h>
#endif

#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"

#include "omniback/core/backend.hpp"
#include "omniback/core/reflect.h"
#include "omniback/core/task_keys.hpp"

#include "tensorrt_torch/tensorrt_helper.hpp"
#include "tensorrt_torch/tensorrt_plugins_anchor.hpp"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h> // 必须包含此头文件
#include <torch/torch.h>
#include "c10/cuda/CUDAGuard.h"
// #include <cuda_runtime.h>

namespace nvinfer1 {
namespace plugin {
#if NV_TENSORRT_MAJOR >= 10
AnchorPlugin::AnchorPlugin(const std::string& params, bool is_build_phase)
    : serialization_(params), is_build_phase_(is_build_phase) {
  params_ = omniback::str::map_split(params, '=', ';');

  omniback::str::try_update(params_, "num_output", anchor_params_.num_output);
  omniback::str::try_update(params_, "num_input", anchor_params_.num_input);
  omniback::str::try_update(params_, "layer_idx", anchor_params_.layer_idx);
  // omniback::str::try_update(params_, "workspace",
  // anchor_params_.workspace_size);
  OMNI_ASSERT(
      anchor_params_.workspace_size <= std::numeric_limits<long int>::max());

  omniback::str::try_update<std::string>(params_, "name", anchor_params_.name);

  std::string dtype = "fp16";
  omniback::str::try_update(params_, "dtype", dtype);
  std::vector<std::string> types = omniback::str::str_split(dtype, ',');
  for (const auto& item : types)
    anchor_params_.type.push_back(torchpipe::convert2trt(item));
  OMNI_ASSERT(!anchor_params_.type.empty());
  if (anchor_params_.num_output + anchor_params_.num_input >
      anchor_params_.type.size()) {
    anchor_params_.type.resize(
        anchor_params_.num_output + anchor_params_.num_input,
        anchor_params_.type.back());
  }

  initFieldsToSerialize();

  [[maybe_unused]] static auto _ = [this]() {
    SPDLOG_INFO("anchor plugin: name={}", anchor_params_.name);
    return 1;
  }();

  if (!is_build_phase) {
    // dependency_ = OMNI_INSTANCE_GET(omniback::Backend, anchor_params_.name);
    dependency_ =
        omniback::init_backend(kANCHOR_PLUGIN_NAME, params_).release();
    OMNI_ASSERT(dependency_);
  }
}

void AnchorPlugin::initFieldsToSerialize() {
  // Serialize AnchorPluginParameters.
  mDataToSerialize.clear();
  mDataToSerialize.emplace_back(
      nvinfer1::PluginField(
          "params",
          serialization_.data(),
          PluginFieldType::kCHAR,
          serialization_.size()));
  mFCToSerialize.nbFields = mDataToSerialize.size();
  mFCToSerialize.fields = mDataToSerialize.data();
}

IPluginCapability* AnchorPlugin::getCapabilityInterface(
    PluginCapabilityType type) noexcept {
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime*>(this);
    }
    OMNI_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore*>(this);
  } catch (std::exception const& e) {
    SPDLOG_ERROR("Got exception: {}", e.what());
  }
  return nullptr;
}

IPluginV3* AnchorPlugin::clone() noexcept {
  // It's possible to encounter errors during cloning.
  // For example, if the memory to allocate is insufficient, exceptions can be
  // thrown.
  try {
    IPluginV3* const plugin{new AnchorPlugin{serialization_, is_build_phase_}};
    return plugin;
  } catch (std::exception const& e) {
    SPDLOG_ERROR("Got exception: {}", e.what());
  }
  return nullptr;
}

char const* AnchorPlugin::getPluginName() const noexcept {
  return kANCHOR_PLUGIN_NAME;
}

char const* AnchorPlugin::getPluginVersion() const noexcept {
  return kANCHOR_PLUGIN_VERSION;
}

char const* AnchorPlugin::getPluginNamespace() const noexcept {
  return kANCHOR_PLUGIN_NAMESPACE;
}

int32_t AnchorPlugin::getNbOutputs() const noexcept {
  // SPDLOG_INFO("getNbOutputs called. num_output = {}",
  // anchor_params_.num_output);
  return anchor_params_.num_output;
  //   return 1;
}

int32_t AnchorPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
  // Communicates the number of inputs and outputs, dimensions, and datatypes
  // of all inputs and outputs, broadcast information for all inputs and
  // outputs, the chosen plugin format, and maximum batch size. At this point,
  // the plugin sets up its internal state and selects the most appropriate
  // algorithm and data structures for the given configuration. Note: Resource
  // allocation is not allowed in this API because it causes a resource leak.

  // This member function will only be called during engine build time.

  // Validate input arguments.
  anchor_params_.num_input = nbInputs;
  anchor_params_.num_output = nbOutputs;

  [[maybe_unused]] static auto _ = [nbInputs, nbOutputs, in, out] {
    SPDLOG_INFO(
        "configurePlugin. nbInputs={}, nbOutputs={}, in[0].desc.dims.nbDims={}, out[0].desc.dims.nbDims={},in[0].desc.type={}",
        nbInputs,
        nbOutputs,
        in[0].desc.dims.nbDims,
        out[0].desc.dims.nbDims,
        int(in[0].desc.type));
    return true;
  }();

  return 0;
}

bool AnchorPlugin::supportsFormatCombination(
    int32_t pos,
    DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  // For this method inputs are numbered 0..(nbInputs-1) and outputs are
  // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
  // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
  // if (anchor_params_.type != inOut[0].desc.type) {
  //   return false;
  // }

  [[maybe_unused]] static auto _ = [nbInputs, nbOutputs, inOut, pos] {
    SPDLOG_INFO(
        "supportsFormatCombination(pos={}) called. nbInputs={}, nbOutputs={}, type={}",
        pos,
        nbInputs,
        nbOutputs,
        int(inOut[pos].desc.type));
    return true;
  }();

  if (inOut[pos].desc.type != anchor_params_.type[pos] ||
      inOut[pos].desc.format != nvinfer1::PluginFormat::kLINEAR) {
    // SPDLOG_INFO("skip type {}", size_t(inOut[pos].desc.type));
    return false;
  }

  return true;
}

int32_t AnchorPlugin::getOutputDataTypes(
    DataType* outputTypes,
    int32_t nbOutputs,
    DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  // One output.
  SPDLOG_INFO(
      "getOutputDataTypes called. nbInputs={}, nbOutputs={}",
      nbInputs,
      nbOutputs);
  // The output type is the same as the input type.
  for (size_t i = 0; i < nbOutputs; ++i) {
    outputTypes[i] = nvinfer1::DataType::kHALF; // inputTypes[0];
  }

  return 0;
}

int32_t AnchorPlugin::getOutputShapes(
    DimsExprs const* inputs,
    int32_t nbInputs,
    DimsExprs const* shapeInputs,
    int32_t nbShapeInputs,
    DimsExprs* outputs,
    int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept {
  // SPDLOG_INFO("getOutputShapes called. ");
  if (nbOutputs < nbInputs) {
    SPDLOG_ERROR(
        "not support : nbOutputs({}) < nbInputs({}). ", nbOutputs, nbInputs);
    return -1;
  }
  for (size_t i = 0; i < nbOutputs; ++i) {
    outputs[i].nbDims = inputs[i].nbDims;
    for (int32_t j{0}; j < outputs[i].nbDims; ++j) {
      outputs[i].d[j] = inputs[i].d[j];
    }
  }

  return 0;
}

int32_t AnchorPlugin::enqueue(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* outputDesc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  [[maybe_unused]] static auto _ = []() {
    SPDLOG_INFO("ENQUEUE ");
    return true;
  }();

  // 获取当前CUDA设备
  int device_id;
  cudaError_t cuda_status = cudaGetDevice(&device_id);
  if (cuda_status != cudaSuccess) {
    return cuda_status;
  }

  // 2. 条件性创建流守卫
  at::cuda::CUDAEvent pre_event;
  std::unique_ptr<c10::cuda::CUDAStreamGuard> guard;
  if (at::cuda::getCurrentCUDAStream(device_id) != stream) {
    SPDLOG_WARN("use External stream");
    guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
        c10::cuda::getStreamFromExternal(stream, device_id));
    pre_event.record(guard->current_stream());
    pre_event.block(guard->original_stream());
  }

  //----------------------------------------------
  // 1. 解析输入张量
  //----------------------------------------------
  const int nbInputs = anchor_params_.num_input;
  std::vector<torch::Tensor> input_tensors;
  for (int i = 0; i < nbInputs; ++i) { // 使用函数参数nbInputs
    const PluginTensorDesc& desc = inputDesc[i];

    // 数据类型映射
    torch::ScalarType dtype;
    switch (desc.type) {
      case DataType::kFLOAT:
        dtype = torch::kFloat32;
        break;
      case DataType::kHALF:
        dtype = torch::kFloat16;
        break;
      case DataType::kINT32:
        dtype = torch::kInt32;
        break;
      case DataType::kINT64:
        dtype = torch::kInt64;
        break;
      case DataType::kINT8:
        dtype = torch::kInt8;
        break;
      default:
        return cudaErrorInvalidValue;
    }

    // 提取维度
    std::vector<int64_t> sizes;
    for (int d = 0; d < desc.dims.nbDims; ++d) {
      sizes.push_back(desc.dims.d[d]);
    }

    // 计算步长
    std::vector<int64_t> strides(desc.dims.nbDims, 1);
    // 连续内存布局
    for (int d = desc.dims.nbDims - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * sizes[d + 1];
    }

    // 创建输入张量视图
    input_tensors.emplace_back(
        torch::from_blob(
            const_cast<void*>(inputs[i]),
            sizes,
            strides,
            torch::TensorOptions().dtype(dtype).device(
                torch::kCUDA, device_id)));
  }

  //----------------------------------------------
  // 2. 构造输出张量容器
  //----------------------------------------------
  const int nbOutputs = anchor_params_.num_output;
  std::vector<torch::Tensor> output_tensors;
  for (int o = 0; o < nbOutputs; ++o) { // 使用函数参数nbOutputs
    const PluginTensorDesc& desc = outputDesc[o];

    // 数据类型映射
    torch::ScalarType dtype;
    switch (desc.type) {
      case DataType::kFLOAT:
        dtype = torch::kFloat32;
        break;
      case DataType::kHALF:
        dtype = torch::kFloat16;
        break;
      case DataType::kINT32:
        dtype = torch::kInt32;
        break;
      case DataType::kINT64:
        dtype = torch::kInt64;
        break;
      case DataType::kINT8:
        dtype = torch::kInt8;
        break;
      default:
        return cudaErrorInvalidValue;
    }

    // 提取维度
    std::vector<int64_t> sizes;
    for (int d = 0; d < desc.dims.nbDims; ++d) {
      sizes.push_back(desc.dims.d[d]);
    }

    // 计算步长
    std::vector<int64_t> strides(desc.dims.nbDims, 1);
    // 连续内存布局
    for (int d = desc.dims.nbDims - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * sizes[d + 1];
    }

    // 创建输出张量视图
    output_tensors.emplace_back(
        torch::from_blob(
            outputs[o],
            sizes,
            strides,
            torch::TensorOptions().dtype(dtype).device(
                torch::kCUDA, device_id)));
  }
  bool in_err = false;
  try {
    if (dependency_) {
      auto io = omniback::make_dict();
      (*io)[omniback::TASK_DATA_KEY] = reinterpret_cast<long long>(stream);
      dependency_->forward({io});
    }
  } catch (const pybind11::error_already_set& e) {
    SPDLOG_ERROR("Python error: {}", e.what());
    // e.restore(); // 保持Python错误状态
    // PyErr_Print(); // 可选：将错误打印到标准错误流
    // return cudaErrorUnknown;
    in_err = true;
  } catch (const std::exception& e) { // 其他C++异常
    SPDLOG_ERROR("C++ runtime error: {}", e.what());
    // return cudaErrorUnknown;
    in_err = true;
  }
  if (in_err)
    return cudaErrorUnknown;

  return cudaGetLastError();
}

int32_t AnchorPlugin::onShapeChange(
    PluginTensorDesc const* in,
    int32_t nbInputs,
    PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
  return 0;
}

IPluginV3* AnchorPlugin::attachToContext(
    IPluginResourceContext* context) noexcept {
  return clone();
}

PluginFieldCollection const* AnchorPlugin::getFieldsToSerialize() noexcept {
  return &mFCToSerialize;
}

size_t AnchorPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
  return anchor_params_.workspace_size;
}
#endif // NV_TENSORRT_MAJOR >= 10
} // namespace plugin
} // namespace nvinfer1

namespace nvinfer1 {
namespace plugin {

#if NV_TENSORRT_MAJOR >= 10
// This is not needed for plugin dynamic registration.
REGISTER_TENSORRT_PLUGIN(AnchorPluginCreator);

// Plugin creator
AnchorPluginCreator::AnchorPluginCreator() {
  // Declare the ONNX attributes that the ONNX parser will collect from the
  // ONNX model that contains the AnchorPlugin node.

  // In our dummy case,
  // attrs={
  //     "params": parameters,
  // }

  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("params", nullptr, PluginFieldType::kUNKNOWN, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

nvinfer1::PluginFieldCollection const* AnchorPluginCreator::
    getFieldNames() noexcept {
  // This is only used in the build phase.
  return &mFC;
}

IPluginV3* AnchorPluginCreator::createPlugin(
    char const* name,
    PluginFieldCollection const* fc,
    TensorRTPhase phase) noexcept {
  // The build phase and the deserialization phase are handled differently.
  if (phase == TensorRTPhase::kBUILD) {
    // The attributes from the ONNX node will be parsed and passed via fc.
    try {
      nvinfer1::PluginField const* fields{fc->fields};
      int32_t nbFields{fc->nbFields};

      OMNI_ASSERT(nbFields >= 1, "nbFields = " + std::to_string(nbFields));

      std::string params;
      for (int32_t i{0}; i < nbFields; ++i) {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "params")) {
          // SPDLOG_INFO("fields[i].length={}", fields[i].length);
          params.resize(size_t(fields[i].length));
          OMNI_ASSERT(
              fields[i].type == nvinfer1::PluginFieldType::kCHAR,
              std::to_string(int(fields[i].type)));
          memcpy(params.data(), fields[i].data, fields[i].length);

        } else {
          SPDLOG_WARN("Skip unkown `attrName` {}", attrName);
        }
      }
      SPDLOG_INFO("Plugin Attributes: params -> {}", params);
      // check
      // [[maybe_unused]] auto map_params = omniback::str::str_split(params,
      // ',');

      AnchorPlugin* const plugin{new AnchorPlugin{params, true}};
      return plugin;
    } catch (std::exception const& e) {
      SPDLOG_ERROR("Got exception: {}", e.what());
    }
    return nullptr;
  } else if (phase == TensorRTPhase::kRUNTIME) {
    // The attributes from the serialized plugin will be passed via fc.
    try {
      nvinfer1::PluginField const* fields{fc->fields};
      int32_t nbFields{fc->nbFields};
      OMNI_ASSERT(nbFields == 1);

      char const* attrName = fields[0].name;
      OMNI_ASSERT(!strcmp(attrName, "params"));
      std::string params;
      OMNI_ASSERT(fields[0].type == nvinfer1::PluginFieldType::kCHAR);
      params.resize(size_t(fields[0].length));
      memcpy(params.data(), fields[0].data, fields[0].length);
      AnchorPlugin* const plugin{new AnchorPlugin{params, false}};
      return plugin;
    } catch (std::exception const& e) {
      SPDLOG_ERROR("Got exception: {}", e.what());
    }
    return nullptr;
  } else {
    return nullptr;
  }
}
#endif // NV_TENSORRT_MAJOR >= 10
} // namespace plugin
} // namespace nvinfer1
