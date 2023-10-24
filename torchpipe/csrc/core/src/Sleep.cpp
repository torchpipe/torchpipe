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

#include "params.hpp"

#include "Backend.hpp"
#include "reflect.h"
#include "base_logging.hpp"
namespace ipipe {

/**
 * @brief 前向时睡眠一定时间
 */
class Sleep : public SingleBackend {
 public:
  /**
   * @param Sleep::time 睡眠时间。默认为10（毫秒）。
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict /*dict_config*/) {
    params_ = std::unique_ptr<Params>(new Params({{"Sleep::time", "10"}, {}}, {}, {}, {}));

    if (!params_->init(config)) return false;
    return true;
  };

  /**
   * @brief 睡眠 Sleep::time 毫秒。
   * @param [out] TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY]
   */
  void forward(dict input_dict) {
    std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(params_->at("Sleep::time"))));
    (*input_dict)[TASK_RESULT_KEY] = (*input_dict)[TASK_DATA_KEY];
  }

 private:
  std::unique_ptr<Params> params_;
};
IPIPE_REGISTER(Backend, Sleep, "Sleep");

}  // namespace ipipe