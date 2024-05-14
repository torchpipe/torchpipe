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

#include <torch/torch.h>
#include "Backend.hpp"
#include "dict.hpp"
#include "reflect.h"

namespace ipipe {

/// @brief 用于测试libtorch内部异常的后端。
class C10Exception : public SingleBackend {
 public:
  /**
   * @exception c10::Error libtorch内部异常。
   */
  void forward(dict) {
    auto options = torch::TensorOptions()
                       .device(torch::kCUDA, -1)
                       .dtype(torch::kByte)  // torch::kByte
                       .layout(torch::kStrided)
                       .requires_grad(false);
    auto input = torch::empty({2, 2, 2}, options, torch::MemoryFormat::Contiguous);

    // auto z =
    input.detach().size(-19090);
  }
};
IPIPE_REGISTER(Backend, C10Exception, "C10Exception");

/// @brief 用于测试异常的后端。
class RuntimeError : public SingleBackend {
 public:
  /**
   * @exception std::runtime_error 用于测试。
   */
  void forward(dict) { throw std::runtime_error("RuntimeError throwed"); }
};
IPIPE_REGISTER(Backend, RuntimeError, "RuntimeError");
}  // namespace ipipe
