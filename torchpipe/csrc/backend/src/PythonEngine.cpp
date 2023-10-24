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

#include "Backend.hpp"
#include "dict.hpp"
#include "reflect.h"
#include "base_logging.hpp"
#ifdef PYBIND
#include <pybind11/stl.h>

// #include <pybind11/pybind11.h>
// #include "tensor_type_caster.hpp"
// #include "torch/extension.h"
namespace py = pybind11;
#endif

namespace ipipe {

class PythonEngine : public Backend {  // todo
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override {
    SPDLOG_ERROR("not implemented");
    return false;
  }

  // forward 按顺序调用， 不需要线程安全
  virtual void forward(const std::vector<dict>&) override {}
  uint32_t max() { return 1; }
};

IPIPE_REGISTER(Backend, PythonEngine, "python");

}  // namespace ipipe
