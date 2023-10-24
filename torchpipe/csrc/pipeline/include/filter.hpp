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

#include "Backend.hpp"
#include "dict.hpp"
#include "reflect.h"
namespace ipipe {

class Filter {
 public:
  enum struct status { Run, Skip, SerialSkip, SubGraphSkip, Break, Error };

  virtual bool init(const std::unordered_map<std::string, std::string>& /*config*/,
                    dict /*dict_config*/) {
    return true;
  };

  virtual status forward(dict input) {
    auto iter = input->find(TASK_RESULT_KEY);
    if (iter == input->end()) {
      // there must be sth wrong

      return status::Break;
    }

    (*input)[TASK_DATA_KEY] = (*input)[TASK_RESULT_KEY];
    input->erase(iter);
    return status::Run;
  }

  virtual ~Filter() = default;
};

class FilterOr : public Filter {
 public:
  status forward(dict input) {
    auto iter = input->find(TASK_RESULT_KEY);
    if (iter == input->end()) return status::Run;
    //(*input)[TASK_DATA_KEY] = (*input)[TASK_RESULT_KEY];
    return status::Skip;
  }
};

template <Filter::status return_status>
class FilterReturn : public Filter {
 public:
  status forward(dict data) { return return_status; }
};

}  // namespace ipipe
