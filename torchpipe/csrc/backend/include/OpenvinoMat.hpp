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

#include "Backend.hpp"
#include "params.hpp"

namespace ipipe {

namespace o_v {
class ModelInstances;
}

class OpenvinoMat : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string> &,
                    dict shared_config) override;

  void forward(const std::vector<dict> &) override;

  /**
   * @brief 此实例中模型的最大batch
   *
   */
  virtual uint32_t max() const { return 1; }
  /**
   * @brief 此实例中模型的最大小batch
   *
   */
  virtual uint32_t min() const { return 1; };
  ~OpenvinoMat(){

  };

 private:
  std::unique_ptr<Params> params_;

  std::unique_ptr<Backend> backend_;

  std::shared_ptr<ipipe::o_v::ModelInstances> instances_;

  int independent_thread_index_{0};
};
}  // namespace ipipe