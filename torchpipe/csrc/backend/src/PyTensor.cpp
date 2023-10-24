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

#include "PyTensor.hpp"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"
// #include "torch/deploy.h"
// #include <torch/csrc/deploy/deploy.h>
#include <fstream>
#include "reflect.h"
#include <ATen/ATen.h>
// #include <torch/extension.h>
#include <torch/python.h>
namespace ipipe {
class PyTensorWrapper {
 public:
  PyTensorWrapper() = default;

  enum class InitState { Failed, Success, Uninit };

  bool init() {
    std::lock_guard<std::mutex> tmp(lock_);
    switch (state_) {
      case (InitState::Failed):
        return false;
      case (InitState::Success):
        return true;
      case (InitState::Uninit):
        state_ = InitState::Failed;
      default:
        break;
    }
    py::gil_scoped_acquire acquire;
    _obj = py::module::import("PyTensor").attr("tensor2tensor");
    state_ = InitState::Success;
    return true;
  }

  ~PyTensorWrapper() { _obj.release(); }

  at::Tensor call(at::Tensor data) {
    py::gil_scoped_acquire acquire;
    // pybind11::object o = pybind11::cast(data);
    // pybind11::handle h = pybind11::cast(data);
    return _obj(data).cast<at::Tensor>();
  }

 private:
  py::object _obj;
  InitState state_{InitState::Uninit};
  std::mutex lock_;
};
PyTensorWrapper gwrapper;

bool PyTensor::init(const std::unordered_map<std::string, std::string>& config_param,
                    dict dict_config) {
  return gwrapper.init();

  // params_ = std::unique_ptr<Params>(new Params({}, {"resize_h", "resize_w"}, {}, {}));
  // if (!params_->init(config_param)) return false;

  // LOG_EXCEPTION(resize_h_ = std::stoi(params_->operator[]("resize_h")));
  // LOG_EXCEPTION(resize_w_ = std::stoi(params_->operator[]("resize_w")));
  // if (resize_h_ > 1024 * 1024 || resize_w_ > 1024 * 1024 || resize_h_ < 1 || resize_w_ < 1) {
  //   SPDLOG_ERROR("PyTensor: illigle h or w: h=" + std::to_string(resize_h_) +
  //                "w=" + std::to_string(resize_w_));
  //   return false;

  return true;
}

void PyTensor::forward(dict input_dict) {
  // params_->check_and_update(input_dict);
  auto& input = *input_dict;
  if (input[TASK_DATA_KEY].type() != typeid(at::Tensor)) {
    SPDLOG_ERROR("PyTensor: error input type: " + std::string(input[TASK_DATA_KEY].type().name()));
    return;
  }
  auto data = any_cast<at::Tensor>(input[TASK_DATA_KEY]);
  data = gwrapper.call(data);
  input[TASK_RESULT_KEY] = data;

  // torch::deploy::InterpreterManager manager(2);

  // // Acquire a session on one of the interpreters
  // auto I = manager.acquireOne();

  // // from builtins import print
  // // print("Hello world!")
  // auto result = I.global("PyTensor", "tensor2tensor")({data});
  // input[TASK_RESULT_KEY] = result;

  return;
}
IPIPE_REGISTER(Backend, PyTensor, "PyTensor");
}  // namespace ipipe