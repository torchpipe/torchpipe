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
#include "Jump.hpp"
namespace ipipe {

class MapReduce : public Jump {
 public:
  bool post_init(const std::unordered_map<std::string, std::string>& config,
                 dict dict_config) override;

  virtual std::vector<dict> split(dict) override;
  virtual void merge(const std::vector<dict>&, dict in_out) override;

 private:
  std::unique_ptr<Params> params_;
  std::string split_;
  std::vector<std::string> merges_;
};
}  // namespace ipipe