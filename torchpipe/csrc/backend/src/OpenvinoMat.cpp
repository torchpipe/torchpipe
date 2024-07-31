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

#include <torch/torch.h>
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

#include <opencv2/core.hpp>

// #include "cnn.hpp"

#include "OvConverter.hpp"
// #include "openvino/openvino.hpp"
// #define USE_OUT_MEM
namespace ipipe {

bool OpenvinoMat::init(const std::unordered_map<std::string, std::string>& config_param,
                       dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"_independent_thread_index", "0"},
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
  IPIPE_ASSERT(instance_num > _independent_thread_index && instance_num > 0);

  std::string model = params_->at("model");
  // get engine
  auto& config = *dict_config;

  // input_shape_ = instances_->.ov_model.input().get_shape();
  // input_shape_[ov::layout::batch_idx({"NHWC"})] = 1;
  std::shared_ptr<model::OvConverter> ov_model;
  if (config.count("_engine") == 0) {
    ov_model = std::make_shared<model::OvConverter>();
    IPIPE_ASSERT(ov_model->init(config_param));

    config["_engine"] = ov_model;
  } else {
    ov_model = any_cast<std::shared_ptr<model::OvConverter>>(config.at("_engine"));
  }
  in_names_ = ov_model->get_input_names();

  out_names_ = ov_model->get_output_names();
  instance_ = ov_model->createInstance();
  IPIPE_ASSERT(instance_);

  assert(config.count("_engine") != 0);
  dict_config_ = dict_config;

  return true;
}
void OpenvinoMat::forward(const std::vector<dict>& input_dicts) {
  cv::Mat input = any_cast<cv::Mat>(input_dicts[0]->at(TASK_DATA_KEY));
  if (input.isContinuous()) {
    input = input.clone();
  }
  IPIPE_ASSERT(input.elemSize1() == 1 || input.elemSize1() == 4);
  std::vector<cv::Mat> outputs;
  IPIPE_ASSERT(in_names_.size() == 1, "only support one input at this time");
  for (std::size_t i = 0; i < in_names_.size(); ++i) {
    instance_->set_input(in_names_[i], input);
  }
  instance_->forward();
  for (std::size_t i = 0; i < out_names_.size(); ++i) {
    cv::Mat output = any_cast<cv::Mat>(instance_->get_output(out_names_[i]));
    outputs.emplace_back(output);
  }

  if (outputs.size() == 1) {
    (*input_dicts[0])[TASK_RESULT_KEY] = outputs[0];
  } else {
    (*input_dicts[0])[TASK_RESULT_KEY] = outputs;
  }

  // https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Preprocessing_Overview.html#doxid-openvino-docs-o-v-u-g-preprocessing-overview
}

IPIPE_REGISTER(Backend, OpenvinoMat, "OpenvinoMat");

}  // namespace ipipe

#endif