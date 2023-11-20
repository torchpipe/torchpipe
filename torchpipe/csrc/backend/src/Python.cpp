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

#include "Python.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"
// #include "torch/deploy.h"
// #include <torch/csrc/deploy/deploy.h>
#include <fstream>
#include "reflect.h"
#include <ATen/ATen.h>
#include "exception.hpp"
#include "tensor_type_caster.hpp"
// #include <torch/extension.h>
#include <torch/python.h>
#include "Backend.hpp"
namespace ipipe {

class RegisterPython {
 public:
  void registe(const std::string& name, py::object obj) {
    std::lock_guard<std::mutex> tmp(lock_);
    objects_[name] = obj;
  }
  py::object create(const std::string& name) {
    std::lock_guard<std::mutex> tmp(lock_);
    auto iter = objects_.find(name);
    if (iter == objects_.end()) {
      throw std::runtime_error("Register of Python: " + name + " not found");
    }
    return objects_[name];
  }

 private:
  static std::unordered_map<std::string, py::object> objects_;
  static std::mutex lock_;
};

std::mutex RegisterPython::lock_ = std::mutex();
std::unordered_map<std::string, py::object> RegisterPython::objects_ =
    std::unordered_map<std::string, py::object>();

void register_py(py::object class_def, const std::string& name) {
  RegisterPython register_python;
  register_python.registe(name, class_def);
}
py::object create_py(const std::string& name) {
  RegisterPython register_python;
  return register_python.create(name);
}

bool Python::init(const std::unordered_map<std::string, std::string>& config_param,
                  dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"max", "1"}}, {"Python::backend"}, {}, {}));

  if (!params_->init(config_param)) {
    return false;
  }
  TRACE_EXCEPTION(max_ = std::stoi(params_->at("max")));

  py::gil_scoped_acquire gil_lock;
  TRACE_EXCEPTION(py_backend_ = create_py(params_->at("Python::backend"))());
  IPIPE_ASSERT(!py_backend_.is(py::none()));
  if (!py_backend_.attr("init")(config_param)) {
    SPDLOG_ERROR(params_->at("Python::backend") + " init failed");
    return false;
  }
  return true;
  // return py_wrapper_->init(params_->at("module_name"), params_->at("backend_name"));
}

void Python::forward(const std::vector<ipipe::dict>& input_dicts) {
  // params_->check_and_update(input_dict);
  py::gil_scoped_acquire gil_lock;
  py::list py_inputs;
  for (const auto& input_dict : input_dicts) {
    py::dict py_input;
    dict2py(input_dict, py_input, true);
    py_inputs.append(py_input);
  }

  py_backend_.attr("forward")(py_inputs);
  std::vector<dict> results;

  for (std::size_t i = 0; i < py::len(py_inputs); ++i) {
    results.push_back(py2dict(py_inputs[i]));
  }

  // update result to input_dict
  for (int i = 0; i < input_dicts.size(); i++) {
    auto result = results[i];
    const auto& input_dict = input_dicts[i];
    for (auto& item : *result) {
      (*input_dict)[item.first] = item.second;
    }
  }

  return;
}

IPIPE_REGISTER(Backend, Python, "Python");
}  // namespace ipipe