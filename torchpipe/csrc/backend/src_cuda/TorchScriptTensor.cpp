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

#include "TorchScriptTensor.hpp"

#include <ATen/ATen.h>

#include <numeric>

#include "base_logging.hpp"
#include "reflect.h"
#include "SingleConcatPreprocess.hpp"
#include "MultipleConcatPreprocess.hpp"
#include "ipipe_common.hpp"
#include "file_utils.hpp"
#include <sstream>
namespace ipipe {

bool TorchScriptTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                             dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"max", "1"}}, {"model"}, {}, {}));
  if (!params_->init(config_param)) return false;
  auto model_path = params_->at("model");
  max_ = std::stoi(params_->at("max"));
  if (!os_path_exists(model_path)) {
    SPDLOG_ERROR("{} not exists", model_path);
    return false;
  }
  try {
    // Deserialize the Scriptmodule from a file using torch::jit::load().
    module_ = torch::jit::load(model_path);
    module_.to(at::kCUDA);  // module_.to(at::Device("cuda:1"));
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/builtin_function.h#L56
    // https://github.com/pytorch/pytorch/blob/01069ad4be449f376cf88a56d842b8eb50f6e9b6/torch/csrc/jit/api/method.h#L23
    num_inputs_ = module_.get_method("forward").num_inputs() - 1;
    IPIPE_ASSERT(num_inputs_ > 0 && num_inputs_ <= 10);

  } catch (const c10::Error& e) {
    SPDLOG_ERROR("error loading the TorchScript model {}", e.what());
    return false;
  }
  if (num_inputs_ == 1)
    preprocessor_ = std::unique_ptr<PreProcessor<at::Tensor>>(new SingleConcatPreprocess());
  else {
    preprocessor_ = std::unique_ptr<PreProcessor<at::Tensor>>(new MultipleConcatPreprocess());
  }
  if (!preprocessor_ || !preprocessor_->init(config_param, dict_config)) {
    std::cout << "preprocess_engine created failed. " << bool(preprocessor_) << std::endl;
    return false;
  }

  std::string batch_post = "split";  // params_->at("postprocessor");

  postprocessor_ = std::unique_ptr<PostProcessor<at::Tensor>>(
      IPIPE_CREATE(PostProcessor<at::Tensor>, batch_post));
  try {
    if (!postprocessor_ || !postprocessor_->init(config_param, dict_config)) {
      SPDLOG_ERROR("error postprocessor: " + batch_post);
      return false;
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("error postprocessor({}): {}", batch_post, e.what());
    return false;
  }

  return true;
}

void TorchScriptTensor::forward(const std::vector<dict>& raw_inputs) {
  auto inputs = preprocessor_->forward(raw_inputs);
  if (inputs.empty()) {
    return;
  }

  std::vector<torch::jit::IValue> model_inputs;
  for (const auto& input : inputs) {
    std::stringstream ss;
    ss << "input.shape " << input.sizes() << std::endl;
    SPDLOG_DEBUG(ss.str());
    if (input.sizes()[0] != raw_inputs.size()) {
      // throw std::runtime_error("only explict batch size supported for TorchScriptTensor");
      SPDLOG_DEBUG("input.sizes()[0] != raw_inputs.size() {} vs {}", input.sizes()[0],
                   raw_inputs.size());
    }
    model_inputs.push_back(input);
  }
  auto out_tmp = module_.forward(model_inputs);

  std::vector<at::Tensor> outputs;
  if (out_tmp.isTensor()) {
    auto out = out_tmp.toTensor();
    IPIPE_ASSERT(out.is_cuda());
    outputs.push_back(out);

  } else if (out_tmp.isTuple()) {
    auto out_data = out_tmp.toTuple();
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13) || (TORCH_VERSION_MAJOR == 2)
    auto num_output = out_data->size();
#else
    auto num_output = out_data->elements().size();
#endif

    for (int i = 0; i < num_output; ++i) {
      outputs.emplace_back(out_data->elements()[i].toTensor());
      // std::cout << "outputs[i].shape" << outputs[i].sizes() << std::endl;
    }
  } else {
    SPDLOG_ERROR("output type not support");
    throw std::runtime_error("output type not support");
  }
  postprocessor_->forward(outputs, raw_inputs, inputs);
}
IPIPE_REGISTER(Backend, TorchScriptTensor, "TorchScriptTensor");

}  // namespace ipipe