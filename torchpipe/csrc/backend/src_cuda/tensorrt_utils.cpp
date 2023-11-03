// Copyright 2021-2023 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorrt_utils.hpp"
#include "base_logging.hpp"
#include "ipipe_common.hpp"
#include <set>
#include <sstream>
namespace ipipe {

bool check_dynamic_batchsize(nvinfer1::INetworkDefinition* network) {
  for (std::size_t index_l = 0; index_l < network->getNbLayers(); ++index_l) {
    nvinfer1::ILayer* layer = network->getLayer(index_l);
    if (layer->getType() == nvinfer1::LayerType::kSHUFFLE) {
      nvinfer1::IShuffleLayer* resizer = static_cast<nvinfer1::IShuffleLayer*>(layer);
      nvinfer1::Dims out = resizer->getReshapeDimensions();
      if (out.nbDims == -1) continue;
      if (out.nbDims == 1) continue;  // Unsqueeze
      static const std::unordered_set<std::string> skip_layers{"ONNXTRT_Broadcast"};
      if (skip_layers.count(resizer->getName())) {
        continue;
      }
      IPIPE_ASSERT(out.nbDims >= 1);
      if (out.d[0] != -1) {
        std::stringstream ss;
        ss << resizer->getName();
        ss << ":\nWhen generating a model with a dynamic batch dimension, it "
              "was "
              "found that the first "
              "dimension of the output shape(which is ";
        for (std::size_t i = 0; i < out.nbDims; ++i) {
          ss << out.d[i];
          if (i != out.nbDims - 1) ss << "x";
        }
        ss << ") of the reshape layer is a fixed integer. Please make "
              "sure this is in line with your expectations. Generally, the first dimension is the "
              "batch dimension and needs to remain dynamic. You may consider changing ‘x.view(1, "
              "…)’ "
              "to ‘x.view(-1, int, int, …)’ to achieve this. ";

        SPDLOG_WARN(colored(ss.str()));
        return false;
      }
    }
  }
  return true;
}

bool precision_fpx_count(const std::set<std::string>& layers, const std::string& name,
                         std::set<std::string>& layers_erased) {
  for (const auto& item : layers) {
    if (name.find(item) != std::string::npos) {
      layers_erased.insert(item);
      return true;
    }
  }
  return false;
}

bool is_ln_name(const std::set<std::string>& layers, const std::string& name) {
  for (const auto& item : layers) {
    if (name.find(item) != std::string::npos) {
      return true;
    }
  }
  return false;
}

// https://github.com/maggiez0138/Swin-Transformer-TensorRT/blob/master/trt/trt_utils.py
void parse_ln(nvinfer1::INetworkDefinition* network) {
  const static std::string POW_NAME = "Pow";
  const static std::string ReduceMean_NAME = "ReduceMean";
  const static std::string Add_NAME = "Add";
  const static std::set<std::string> LN_NAME{POW_NAME, ReduceMean_NAME, Add_NAME, "Sqrt"};

  for (std::size_t index_l = 0; index_l < network->getNbLayers(); ++index_l) {
    nvinfer1::ILayer* layer = network->getLayer(index_l);
    // nvinfer1::ILayer* next_layer = nullptr;
    // nvinfer1::ILayer* previous_layer = nullptr;
    // if (index_l != network->getNbLayers() - 1) next_layer = network->getLayer(index_l + 1);
    // if (index_l != 0) previous_layer = network->getLayer(index_l - 1);
    IPIPE_ASSERT(layer);
    std::string start_name = layer->getName();
    std::vector<nvinfer1::ILayer*> target;
    bool find_ln = false;
    if (start_name.find(POW_NAME) != std::string::npos &&
        !network->getLayer(index_l)->precisionIsSet()) {
      while (index_l < network->getNbLayers()) {
        target.push_back(network->getLayer(index_l));
        if (target.size() >= 6) break;
        ++index_l;
        std::string name = network->getLayer(index_l)->getName();
        if (name.find("Sqrt") != std::string::npos) {
          target.push_back(network->getLayer(index_l));
          find_ln = true;
          break;
        }
      }
    }
    if (find_ln) {
      SPDLOG_INFO("\nLayerNorm matched:");
      for (std::size_t index = 0; index < target.size(); ++index) {
        std::string name = target[index]->getName();
        if (!is_ln_name(LN_NAME, name)) continue;
        target[index]->setPrecision(nvinfer1::DataType::kFLOAT);

        if (index != target.size() - 1) {
          target[index]->setOutputType(0, nvinfer1::DataType::kFLOAT);
          SPDLOG_INFO("{} and its output was set to fp32 mode", target[index]->getName());
        } else {
          SPDLOG_INFO("{} was set to fp32 mode", target[index]->getName());
        }
      }
    }
  }
}
void modify_layers_precision(std::set<std::string> precision_fpx,
                             nvinfer1::INetworkDefinition* network, nvinfer1::DataType dataType,
                             bool is_output) {
  std::set<std::string> layers_erased;
  for (std::size_t index_l = 0; !precision_fpx.empty() && index_l < network->getNbLayers();
       ++index_l) {
    nvinfer1::ILayer* layer = network->getLayer(index_l);
    // nvinfer1::ILayer* next_layer = nullptr;
    // nvinfer1::ILayer* previous_layer = nullptr;
    // if (index_l != network->getNbLayers() - 1) next_layer = network->getLayer(index_l + 1);
    // if (index_l != 0) previous_layer = network->getLayer(index_l - 1);
    IPIPE_ASSERT(layer);
    // if (precision_fpx.count(layer->getName()) != 0) {
    if (precision_fpx_count(precision_fpx, std::string(layer->getName()), layers_erased)) {
      std::string mode_name = "fp32";
      switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
          mode_name = "fp32";
          break;
        case nvinfer1::DataType::kINT8:
          mode_name = "int8";
          break;
        case nvinfer1::DataType::kHALF:
          mode_name = "fp16";
          break;
        default:
          throw std::runtime_error("unsupported data type: " + std::to_string(int(dataType)));
          break;
      }
      if (is_output) {
        layer->setOutputType(0, dataType);
        SPDLOG_INFO("{}'s output was set to {} mode", layer->getName(), mode_name);
      } else {
        layer->setPrecision(dataType);
        SPDLOG_INFO("{} was set to {} mode", layer->getName(), mode_name);
      }

      // layer->setOutputType(0, dataType);
      // previous_layer->setOutputType(0, dataType);

      // precision_fpx.erase(layer->getName());
    }
  }
  for (const auto& item : layers_erased) {
    precision_fpx.erase(item);
  }

  if (!precision_fpx.empty()) {
    std::string error_msg = "The following layers were not found in network: ";
    for (const auto& layers_err : precision_fpx) {
      error_msg += layers_err + ' ';
    }
    error_msg += "-----------------\nExisting layers: ";
    for (std::size_t index_l = 0; !precision_fpx.empty() && index_l < network->getNbLayers();
         ++index_l) {
      error_msg += std::string(network->getLayer(index_l)->getName()) + "\n";
    }
    throw std::runtime_error(error_msg);
  }
}

nvinfer1::ITensor* MeanStd(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
                           float* mean, float* std, std::set<nvinfer1::ILayer*>& new_layers,
                           bool set_half) {
  bool in_error = false;

  for (int i = 0; i < 3; ++i) {
    if (mean && mean[i] <= 1 + 1e-5) {
      in_error = true;
    }

    if (std && std[i] <= 1 + 1e-5) {
      in_error = true;
    }
  }
  if (in_error) {
    throw std::invalid_argument(
        "Input mean or std <= 1+1e-5. In this case, we do not support automatically integrating "
        "the "
        "normalization operation into TensorRT. It should be removed, scaled up by a factor of "
        "255, or add these regularization operations when defining the model.");
  }

  nvinfer1::ITensor* itensor = input;
  if (mean) {
    nvinfer1::Weights Mean{nvinfer1::DataType::kFLOAT, nullptr, 3};
    Mean.values = mean;
    nvinfer1::IConstantLayer* m = network->addConstant(nvinfer1::Dims4{1, 3, 1, 1}, Mean);
    new_layers.insert(m);

    auto* sub_mean =
        network->addElementWise(*itensor, *m->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
    new_layers.insert(sub_mean);
    if (set_half) {
      m->setPrecision(nvinfer1::DataType::kHALF);
      m->setOutputType(0, nvinfer1::DataType::kHALF);
      sub_mean->setPrecision(nvinfer1::DataType::kHALF);
      if (std) sub_mean->setOutputType(0, nvinfer1::DataType::kHALF);
    }
    itensor = sub_mean->getOutput(0);
  }

  if (std) {
    nvinfer1::Weights Std{nvinfer1::DataType::kFLOAT, nullptr, 3};
    Std.values = std;
    nvinfer1::IConstantLayer* s = network->addConstant(nvinfer1::Dims4{1, 3, 1, 1}, Std);
    nvinfer1::IElementWiseLayer* std_mean =
        network->addElementWise(*itensor, *s->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
    new_layers.insert(s);
    new_layers.insert(std_mean);
    if (set_half) {
      s->setPrecision(nvinfer1::DataType::kHALF);
      s->setOutputType(0, nvinfer1::DataType::kHALF);
      std_mean->setPrecision(nvinfer1::DataType::kHALF);
      // std_mean->setOutputType(0, nvinfer1::DataType::kHALF);
    }
    itensor = std_mean->getOutput(0);
  }
  return itensor;
}

bool is_qat(nvinfer1::INetworkDefinition* network) {
  for (std::size_t index_l = 0; index_l < network->getNbLayers(); ++index_l) {
    nvinfer1::ILayer* layer = network->getLayer(index_l);
    auto type = layer->getType();
    SPDLOG_DEBUG("NV_TENSORRT_MAJOR defined {}", NV_TENSORRT_MAJOR);
#if NV_TENSORRT_MAJOR >= 8
    if (type == nvinfer1::LayerType::kQUANTIZE) {
      SPDLOG_INFO("found (DE)QUANTIZE layer. Use QAT mode");
      return true;
    }
#endif
  }
  return false;
}

}  // namespace ipipe