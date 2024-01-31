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

#ifdef WITH_OPENVINO

#include <ATen/ATen.h>
#include <fstream>
#include <memory>
#include <numeric>
#include <thread>
// #include "DynamicOnnx2TrtBackend.hpp"
#include "base_logging.hpp"
#include "Backend.hpp"
#include "dict.hpp"
// #include "hw_batching.hpp"

#include "reflect.h"
#include "time_utils.hpp"
#include "torch_utils.hpp"
#include "base_logging.hpp"
#include "exception.hpp"
#include "OpenvinoMat.hpp"
#include <openvino/openvino.hpp>

// https://github.com/NVIDIA/Torch-TensorRT/blob/3a98a8b198a071e622c43283caea7416fe8a8a1a/core/runtime/register_trt_op.cpp

// #define USE_OUT_MEM
namespace ipipe {

struct ModelInstances {
  ov::CompiledModel compiled_model;
  std::vector<ov::InferRequest> ireqs;
};

bool OpenvinoMat::init(const std::unordered_map<std::string, std::string>& config_param,
                       dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"postprocessor", "split"},
                                                {"preprocessor", ""},
                                                {"_independent_thread_index", "0"},
                                                {"OpenvinoMat::backend", ""},
                                                {"instance_num", "1"},
                                                {"precision", "fp32"}},
                                               {"model"}, {}, {}));

  if (!params_->init(config_param)) return false;
  if (!dict_config) {
    dict_config = std::make_shared<std::unordered_map<std::string, any>>();
  }

  int _independent_thread_index = 0;
  // set tensorrt profile
  if (!params_->at("_independent_thread_index").empty()) {
    TRACE_EXCEPTION(_independent_thread_index =
                        std::stoi(params_->at("_independent_thread_index")));
  } else {
    _independent_thread_index = 0;
  }

  independent_thread_index_ = _independent_thread_index;
  int instance_num = std::stoi(params_->at("instance_num"));
  if (instance_num <= _independent_thread_index) {
    SPDLOG_ERROR("instance_num <= _independent_thread_index: " + std::to_string(instance_num) +
                 " <= " + std::to_string(_independent_thread_index));
    return false;
  }

  std::string model = params_->at("model");
  // get engine
  auto& config = *dict_config;
  ov::Core core;
  // core.set_property("CPU", ov::streams::num(ov::streams::AUTO));

  instances_ = std::make_shared<ModelInstances>();

  if (config.count("_engine") == 0) {
    instances_->compiled_model = core.compile_model(
        model, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
        // ov::streams::num(instance_num), ov::inference_num_threads(instance_num),
        ov::hint::num_requests(instance_num), ov::hint::allow_auto_batching(false));

    auto nthreads = instances_->compiled_model.get_property(ov::inference_num_threads);
    // IPIPE_ASSERT(nthreads == instance_num);

    auto nireq = instances_->compiled_model.get_property(ov::optimal_number_of_infer_requests);

    if (nireq != instance_num) {
      SPDLOG_WARN(
          "optimal number of requests does not match the `instance_num`. It is "
          "recommended to set the number of instances equal to the recommended optimal number of "
          "requests");
    }

    // /* Take information about all topology inputs */
    // auto inputs = model->inputs();
    // /* Take information about all topology outputs */
    // auto outputs = model->outputs();
    // ov::Shape static_shape;
    // ov::PartialShape partial_shape =
    //     node->output(0).get_partial_shape();  // get zero output partial shape
    // if (!partial_shape.is_dynamic() /* or partial_shape.is_static() */) {
    //   static_shape = partial_shape.get_shape();
    // }
    config["_engine"] = instances_;
  } else {
    instances_ = any_cast<std::shared_ptr<ModelInstances>>(config.at("_engine"));
  }
  IPIPE_ASSERT(instances_);

  assert(config.count("_engine") != 0);

  std::string precision = params_->at("precision");

  return true;
}
void OpenvinoMat::forward(const std::vector<dict>&) {}

IPIPE_REGISTER(Backend, OpenvinoMat, "OpenvinoMat");

}  // namespace ipipe

#endif