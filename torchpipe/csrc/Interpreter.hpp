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

#pragma once

#include "config_parser.hpp"
#include "reflect.h"
#include <mutex>

#ifdef PYBIND
#include <pybind11/pybind11.h>
#include "tensor_type_caster.hpp"

#endif

namespace ipipe {

class Interpreter : public Backend {
 public:
  void init(mapmap config);
  Interpreter(mapmap config) { init(config); }
  Interpreter() = default;

  void init_from_toml(std::string toml_path) {
    auto config = parse_toml(toml_path);

    init(config);
  }

  void operator()(dict data) { forward({data}); }

  bool init(const std::unordered_map<std::string, std::string>& config,
            dict shared_config) override;

  Interpreter(const std::unordered_map<std::string, std::string>& config) { init(config); }
  void init(const std::unordered_map<std::string, std::string>& config);

  void forward(const std::vector<dict>& input_dicts) override;

  uint32_t max() const override;
  uint32_t min() const override;

#ifdef PYBIND  // python API

  void forward(py::list py_inputs) {
    std::vector<dict> inputs;

    for (std::size_t i = 0; i < py::len(py_inputs); ++i) {
      inputs.push_back(py2dict(py_inputs[i]));
    }

    {
      py::gil_scoped_release gil_lock;
      forward(inputs);
    }

    for (std::size_t i = 0; i < py::len(py_inputs); ++i) {
      dict2py(inputs[i], py_inputs[i]);
    }

    return;
  }

  void forward(py::dict py_input);
  ~Interpreter() {
    if (PyGILState_Check()) {
      py::gil_scoped_release gil_lock;  // https://github.com/pybind/pybind11/issues/1446
      backend_.reset();
    }
  }

#endif

 private:
  void env_init(const std::unordered_map<std::string, std::string>& config, dict /*dict_config*/,
                const std::string& env, bool& finished);

  std::unique_ptr<Backend> backend_;

  static std::once_flag once_flag_;
};

}  // namespace ipipe