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

#pragma once

#include <string>
#include <unordered_map>
#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"
#include <chrono>

namespace ipipe {
/**
 * @brief 获取当前解释器，并将任务转发到解释器执行。
 *
 */
class Jump : public Backend {
 public:
  virtual bool post_init(const std::unordered_map<std::string, std::string>& config,
                         dict dict_config) {
    return true;
  }
  virtual std::vector<dict> split(dict);
  virtual void merge(const std::vector<dict>&, dict in_out);

  bool init(const std::unordered_map<std::string, std::string>& config,
            dict dict_config) override final;
  virtual void forward(const std::vector<dict>& input_dicts) override final;

  // void forward(dict);

  uint32_t max() const override final { return 1; };

 protected:
  std::string jump_;
  Backend* interpreter_;

 private:
  std::vector<dict> split_wrapper(dict);
  void merge_wrapper(dict, const std::vector<dict>& split_data);
};
}  // namespace ipipe