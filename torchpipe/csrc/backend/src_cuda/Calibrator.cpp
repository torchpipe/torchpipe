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

#include "Calibrator.hpp"
#include <iterator>
#include <fstream>
#include "base_logging.hpp"
#include "file_utils.hpp"
#include "reflect.h"
#include "Backend.hpp"
#include "torch_utils.hpp"
#include "exception.hpp"

namespace ipipe {

int Calibrator::getBatchSize() const noexcept { return batchsize_; }

bool Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
  if (read_batch_index_ >= max_batch_num_) {
    return false;
  }

  if (nbBindings != 1) {
    SPDLOG_ERROR("Calibrator: only support nbBindings == 1 at this time.");
    return false;
  }

  std::vector<at::Tensor> input_imgs_;
  for (int i = read_batch_index_ * batchsize_; i < (1 + read_batch_index_) * batchsize_; i++) {
    const auto img_index_ = i % files_.size();
    SPDLOG_INFO("load {}. batch_index = {}, file_index = {}", files_[img_index_], read_batch_index_,
                img_index_);
    auto data = ipipe::load_tensor(files_[img_index_]).cuda();
    input_imgs_.push_back(data);
  }
  at::Tensor final_tensor;
  if (input_imgs_.size() == 1) {
    final_tensor = input_imgs_[0];
  } else {
    final_tensor = at::cat(input_imgs_, 0);
  }

  assert(final_tensor.size(0) == batchsize_);
  final_tensor = final_tensor.to(at::kCUDA, at::kFloat, false, false);
  if (!final_tensor.is_contiguous()) final_tensor = final_tensor.contiguous();

  bindings[0] = final_tensor.data_ptr<float>();
  read_batch_index_++;
  return true;
}

const void* Calibrator::readCalibrationCache(size_t& length) noexcept {
  if (!calib_table_name_.empty() && os_path_exists(calib_table_name_)) {
    SPDLOG_WARN("reading calib cache: [{}]. {}", calib_table_name_,
                "If regeneration is needed, delete it.");
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (input.good()) {
      std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
  } else {
    length = 0;
    return nullptr;
  }
}

void Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
  if (!calib_table_name_.empty()) {
    SPDLOG_INFO("writing calib cache: {} size: {}", calib_table_name_, length);
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
  }
}

bool Calibrator::init(const std::unordered_map<std::string, std::string>& config_param,
                      dict dict_config) {
  params_ = std::unique_ptr<Params>(new Params({{"calibrate_cache", ""}, {"calibrate_input", ""}},
                                               {"calibrate_batchsize"}, {}, {}));
  if (!params_->init(config_param)) return false;
  TRACE_EXCEPTION(batchsize_ = std::stoi(params_->at("calibrate_batchsize")));
  IPIPE_ASSERT(batchsize_ > 0 && batchsize_ < 1e5);
  calib_table_name_ = params_->at("calibrate_cache");
  if (os_path_exists(calib_table_name_)) {
    SPDLOG_INFO("calib table name : {}", calib_table_name_);
    return true;
  }
  auto calibration_tensor_dir = params_->at("calibrate_input");
  if (!os_path_exists(calibration_tensor_dir)) {
    SPDLOG_ERROR("calibrate_input not setted or not exists. {}. ", calibration_tensor_dir);
    return false;
  }
  files_ = os_listdir(calibration_tensor_dir.c_str(), true);

  if (files_.empty()) {
    SPDLOG_ERROR("{}: no data found", calibration_tensor_dir);
    return false;
  }
  if (files_.size() < 20) {
    SPDLOG_WARN("found {} files. It`s better to use 100 - 1000 files.", files_.size());
  }
  max_batch_num_ = files_.size() / batchsize_ + 1;
  if (max_batch_num_ * batchsize_ > 1000) max_batch_num_ = 1000 / batchsize_;
  return true;
}

}  // namespace ipipe