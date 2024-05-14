// Copyright 2021-2024 NetEase.
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

#include "CreateTensor.hpp"

#include <torch/torch.h>

#include <numeric>

#include "base_logging.hpp"
#include "reflect.h"
#include "torch_utils.hpp"
#include "exception.hpp"

namespace ipipe {

bool CreateTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                        dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"type", "byte"}}, {"shape"}, {}, {}));
  if (!params_->init(config_param)) return false;

  TRACE_EXCEPTION(shape_ = str2number<long int>(params_->at("shape"), ','));
  IPIPE_ASSERT(!shape_.empty());
  //   index_selecter_cpu_ = torch::Tensor({2, 1, 0}).toType(torch::kLong);
  //   index_selecter_ = torch::Tensor({2, 1, 0}).toType(torch::kLong).cuda();

  type_ = params_->at("type");
  if (type_ == "char") type_ = "byte";
  IPIPE_ASSERT(type_ == "byte" || type_ == "float");
  return true;
}

void CreateTensor::forward(dict input_dict) {
  static const auto type = (type_ == "byte") ? torch::kByte : torch::kFloat;
  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, -1)
                     .dtype(type)
                     .layout(torch::kStrided)
                     .requires_grad(false);
  auto image_tensor =
      torch::empty(torch::IntArrayRef(shape_), options, torch::MemoryFormat::Contiguous);

  (*input_dict)[TASK_RESULT_KEY] = image_tensor;
}
IPIPE_REGISTER(Backend, CreateTensor, "CreateTensor");

}  // namespace ipipe