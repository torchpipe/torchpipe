// midified from: https://github.com/leimao/TensorRT-Custom-Plugin-Example

#include "TorchPlugin.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <vector>
#include "Interpreter.hpp"
#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <torch/torch.h>

#define PLUGIN_ASSERT(val) reportAssertion((val), #val, __FILE__, __LINE__)

namespace {
constexpr std::size_t knbInputs = 3;
void caughtError(std::exception const& e) {
  getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());
}

void logInfo(char const* msg) { getLogger()->log(nvinfer1::ILogger::Severity::kINFO, msg); }
void logError(char const* msg) { getLogger()->log(nvinfer1::ILogger::Severity::kERROR, msg); }

void reportAssertion(bool success, char const* msg, char const* file, int32_t line) {
  if (!success) {
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << std::endl
           << file << ':' << line << std::endl
           << "Aborting..." << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    std::abort();
  }
}
}  // namespace

namespace {
torch::ScalarType getTorchTypeFromTrtType(nvinfer1::DataType trttype) {
  const static std::map<nvinfer1::DataType, torch::ScalarType> trttype2torchtype = {
      {nvinfer1::DataType::kINT8, torch::kInt8},   {nvinfer1::DataType::kHALF, torch::kHalf},
      {nvinfer1::DataType::kFLOAT, torch::kFloat}, {nvinfer1::DataType::kINT32, torch::kInt32},
      {nvinfer1::DataType::kBOOL, torch::kBool},
  };
  return trttype2torchtype.at(trttype);
}

void io_tensor(nvinfer1::PluginTensorDesc const* inputDesc,
               nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
               void* const* outputs, std::vector<torch::Tensor>& input_arrays,
               std::vector<torch::Tensor>& output_arrays, nvinfer1::DataType trttype) noexcept {
  for (std::size_t i{0}; i < knbInputs; ++i) {
    std::vector<int64_t> sizes;
    for (int j = 0; j < inputDesc[i].dims.nbDims; j++) {
      sizes.push_back(inputDesc[i].dims.d[j]);
    }
    auto options =
        torch::TensorOptions().dtype(getTorchTypeFromTrtType(trttype)).device(torch::kCUDA);
    input_arrays.emplace_back(torch::from_blob((void*)inputs[i], sizes, options));
  }

  for (std::size_t i{0}; i < 1; ++i) {
    std::vector<int64_t> sizes;
    for (int j = 0; j < outputDesc[i].dims.nbDims; j++) {
      sizes.push_back(outputDesc[i].dims.d[j]);
    }
    auto options =
        torch::TensorOptions().dtype(getTorchTypeFromTrtType(trttype)).device(torch::kCUDA);
    output_arrays.emplace_back(torch::from_blob((void*)outputs[i], sizes, options));
  }
}
}  // namespace

namespace nvinfer1 {
namespace plugin {

// Plugin
TorchPlugin::TorchPlugin(TorchPluginParameters const& params) : mParams{params} {
  initFieldsToSerialize();
}

void TorchPlugin::initFieldsToSerialize() {
  // Serialize TorchPluginParameters.
  mDataToSerialize.clear();
  mDataToSerialize.emplace_back(nvinfer1::PluginField(
      "parameters", &mParams, PluginFieldType::kUNKNOWN, sizeof(TorchPluginParameters)));
  mFCToSerialize.nbFields = mDataToSerialize.size();
  mFCToSerialize.fields = mDataToSerialize.data();
}

IPluginCapability* TorchPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild*>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime*>(this);
    }
    PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore*>(this);
  } catch (std::exception const& e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV3* TorchPlugin::clone() noexcept {
  // It's possible to encounter errors during cloning.
  // For example, if the memory to allocate is insufficient, exceptions can be
  // thrown.
  try {
    IPluginV3* const plugin{new TorchPlugin{mParams}};
    return plugin;
  } catch (std::exception const& e) {
    caughtError(e);
  }
  return nullptr;
}

char const* TorchPlugin::getPluginName() const noexcept { return kTORCH_PLUGIN_NAME; }

char const* TorchPlugin::getPluginVersion() const noexcept { return kTORCH_PLUGIN_VERSION; }

char const* TorchPlugin::getPluginNamespace() const noexcept { return kTORCH_PLUGIN_NAMESPACE; }

int32_t TorchPlugin::getNbOutputs() const noexcept { return 1; }

int32_t TorchPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
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
  PLUGIN_ASSERT(
      nbInputs ==
      knbInputs);  // query_states:1x244x2560,key_states:1x244x2560,value_states:1x244x2560
  PLUGIN_ASSERT(nbOutputs == 1);  // attn_output:1x244x2560
  PLUGIN_ASSERT(in[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(in[0].desc.dims.d[0] == out[0].desc.dims.d[0]);
  PLUGIN_ASSERT(in[0].desc.dims.d[1] == out[0].desc.dims.d[1]);
  // PLUGIN_ASSERT(in[0].desc.dims.d[2] == out[0].desc.dims.d[2]);
  PLUGIN_ASSERT(in[0].desc.type == out[0].desc.type);

  mParams.dtype = in[0].desc.type;
  mParams.channelSize = in[0].desc.dims.d[0];
  mParams.height = in[0].desc.dims.d[1];
  mParams.width = in[0].desc.dims.d[2];

  if (mParams.dtype == nvinfer1::DataType::kINT8) {
    mParams.dtypeBytes = 1;
  } else if (mParams.dtype == nvinfer1::DataType::kHALF) {
    mParams.dtypeBytes = 2;
  } else if (mParams.dtype == nvinfer1::DataType::kFLOAT) {
    mParams.dtypeBytes = 4;
  } else {
    PLUGIN_ASSERT(false);
  }

  return 0;
}

bool TorchPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                            int32_t nbInputs, int32_t nbOutputs) noexcept {
  // For this method inputs are numbered 0..(nbInputs-1) and outputs are
  // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
  // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
  PLUGIN_ASSERT(nbInputs == knbInputs && nbOutputs == 1 && pos < nbInputs + nbOutputs);
  bool isValidCombination = false;

  // Suppose we support only a limited number of format configurations.
  //   isValidCombination |= (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
  //                          inOut[pos].desc.type == nvinfer1::DataType::kFLOAT);
  isValidCombination |= (inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
                         inOut[pos].desc.type == nvinfer1::DataType::kHALF);
  // Make sure the input tensor and output tensor types and formats are same.
  isValidCombination &= (pos < nbInputs || (inOut[pos].desc.format == inOut[0].desc.format &&
                                            inOut[pos].desc.type == inOut[0].desc.type));

  return isValidCombination;
}

int32_t TorchPlugin::getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                                        DataType const* inputTypes,
                                        int32_t nbInputs) const noexcept {
  // One output.
  PLUGIN_ASSERT(nbInputs == knbInputs);
  PLUGIN_ASSERT(nbOutputs == 1);
  // The output type is the same as the input type.
  outputTypes[0] = inputTypes[0];
  return 0;
}

int32_t TorchPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                                     DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                                     DimsExprs* outputs, int32_t nbOutputs,
                                     IExprBuilder& exprBuilder) noexcept {
  PLUGIN_ASSERT(nbInputs == knbInputs);
  PLUGIN_ASSERT(nbOutputs == 1);
  PLUGIN_ASSERT(inputs != nullptr);
  PLUGIN_ASSERT(inputs[0].nbDims == 2);

  outputs[0].nbDims = inputs[0].nbDims;
  for (int32_t i{0}; i < inputs[0].nbDims; ++i) {
    outputs[0].d[i] = inputs[0].d[i];
  }

  return 0;
}

int32_t TorchPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                             void const* const* inputs, void* const* outputs, void* workspace,
                             cudaStream_t stream) noexcept {
  {
    std::vector<torch::Tensor> input_arrays;
    std::vector<torch::Tensor> output_arrays;

    io_tensor(inputDesc, outputDesc, inputs, outputs, input_arrays, output_arrays, mParams.dtype);

    // for (auto& item : input_arrays) {
    //   if (item.sizes().size() == 2) item = item.unsqueeze(0);
    // }
    // for (auto& item : output_arrays) {
    //   if (item.sizes().size() == 2) item = item.unsqueeze(0);
    // }

    // Interrupt torch's cuda semantics
    auto ret = cudaStreamSynchronize(stream);
    // throw std::runtime_error("debug here in TorchPlugin");
    assert(cudaSuccess == 0);
    if (ret != cudaSuccess) return ret;
    // IPEIPE_ASSERT(c10::cuda::getCurrentCUDAStream(-1).stream == stream);

    auto inter = ipipe::CThreadSafeInterpreters::getInstance().get();
    assert(inter.size() > 0);
    const auto index = 0;
    // std::unordered_map<std::string, ipipe::any> usr_data = ;

    ipipe::dict user_data = std::make_shared<std::unordered_map<std::string, ipipe::any>>(
        std::unordered_map<std::string, ipipe::any>({{"data", input_arrays},
                                                     {"outputs", output_arrays},
                                                     {"node_name", std::string("TorchPlugin")}}));
    try {
      inter[index]->forward({user_data});
    } catch (std::exception const& e) {
      caughtError(e);
      return -1;
    }

    if (user_data->find("result") == user_data->end()) {
      logError("result not found in user_data");
      return -1;
    }

    // for (auto& item : output_arrays) {
    //   if (item.sizes().size() == 3) item = item.squeeze(0);
    // }

    std::cout << output_arrays[0].sizes()
              << output_arrays[0].index({torch::indexing::Slice(torch::indexing::None, 4),
                                         torch::indexing::Slice(torch::indexing::None, 4)})
              << std::endl;

    // return 0;
  }
  // size_t const inputSize{static_cast<size_t>(inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1])};
  // size_t const inputSizeBytes{inputSize * mParams.dtypeBytes};
  // cudaError_t const status{
  //     cudaMemcpyAsync(outputs[0], inputs[0], inputSizeBytes, cudaMemcpyDeviceToDevice, stream)};
  return 0;
}

int32_t TorchPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInputs,
                                   PluginTensorDesc const* out, int32_t nbOutputs) noexcept {
  return 0;
}

IPluginV3* TorchPlugin::attachToContext(IPluginResourceContext* context) noexcept {
  return clone();
}

PluginFieldCollection const* TorchPlugin::getFieldsToSerialize() noexcept {
  return &mFCToSerialize;
}

size_t TorchPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                     DynamicPluginTensorDesc const* outputs,
                                     int32_t nbOutputs) const noexcept {
  return 0;
}

// PluginCreator

// This is not needed for plugin dynamic registration.
REGISTER_TENSORRT_PLUGIN(TorchPluginCreator);

// Plugin creator
TorchPluginCreator::TorchPluginCreator() {
  // Declare the ONNX attributes that the ONNX parser will collect from the
  // ONNX model that contains the TorchPlugin node.

  // In our dummy case,
  // attrs={
  //     "kernel_shape": [1, 1],
  //     "strides": [1, 1],
  //     "pads": [0, 0, 0, 0],
  //     "group": num_groups
  // }

  // mPluginAttributes.clear();
  // mPluginAttributes.emplace_back(
  //     nvinfer1::PluginField("kernel_shape", nullptr, PluginFieldType::kINT32, 2));
  // mPluginAttributes.emplace_back(
  //     nvinfer1::PluginField("strides", nullptr, PluginFieldType::kINT32, 2));
  // mPluginAttributes.emplace_back(
  //     nvinfer1::PluginField("pads", nullptr, PluginFieldType::kINT32, 4));
  // mPluginAttributes.emplace_back(
  //     nvinfer1::PluginField("group", nullptr, PluginFieldType::kINT32, 1));

  // mFC.nbFields = mPluginAttributes.size();
  // mFC.fields = mPluginAttributes.data();
  mFC.fields = nullptr;
  mFC.nbFields = 0;
}

nvinfer1::PluginFieldCollection const* TorchPluginCreator::getFieldNames() noexcept {
  // This is only used in the build phase.
  return &mFC;
}

IPluginV3* TorchPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc,
                                            TensorRTPhase phase) noexcept {
  // The build phase and the deserialization phase are handled differently.
  if (phase == TensorRTPhase::kBUILD) {
    // The attributes from the ONNX node will be parsed and passed via fc.
    try {
      nvinfer1::PluginField const* fields{fc->fields};
      int32_t nbFields{fc->nbFields};

      PLUGIN_ASSERT(nbFields == 0);

      TorchPluginParameters const params{};

      TorchPlugin* const plugin{new TorchPlugin{params}};
      return plugin;
    } catch (std::exception const& e) {
      caughtError(e);
    }
    return nullptr;
  } else if (phase == TensorRTPhase::kRUNTIME) {
    // The attributes from the serialized plugin will be passed via fc.
    try {
      nvinfer1::PluginField const* fields{fc->fields};
      int32_t nbFields{fc->nbFields};
      PLUGIN_ASSERT(nbFields == 1);

      char const* attrName = fields[0].name;
      PLUGIN_ASSERT(!strcmp(attrName, "parameters"));
      PLUGIN_ASSERT(fields[0].type == nvinfer1::PluginFieldType::kUNKNOWN);
      PLUGIN_ASSERT(fields[0].length == sizeof(TorchPluginParameters));
      TorchPluginParameters params{*(static_cast<TorchPluginParameters const*>(fields[0].data))};

      TorchPlugin* const plugin{new TorchPlugin{params}};
      return plugin;
    } catch (std::exception const& e) {
      caughtError(e);
    }
    return nullptr;
  } else {
    return nullptr;
  }
}

}  // namespace plugin
}  // namespace nvinfer1

// extern "C" nvinfer1::IPluginCreatorInterface* const* getPluginCreators(int32_t& nbCreators) {
//   nbCreators = 1;
//   static nvinfer1::plugin::TorchPluginCreator torchPluginCreator{};
//   static nvinfer1::IPluginCreatorInterface* const pluginCreatorList[] = {&torchPluginCreator};
//   return pluginCreatorList;
// }