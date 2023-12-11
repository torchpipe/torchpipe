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

#include "filter.hpp"
#include "params.hpp"
namespace ipipe {
IPIPE_REGISTER(Filter, Filter, "swap")

IPIPE_REGISTER(Filter, FilterOr, "or");
// IPIPE_REGISTER(Filter, FilterOr, "FilterOr");
// using FilterReturn_Filter_status_skip_ = FilterReturn<Filter::status::Skip>;
// IPIPE_REGISTER(Filter, FilterReturn_Filter_status_skip_, "Skip");

using FilterReturn_Filter_status_Skip_ = FilterReturn<Filter::status::Skip>;
IPIPE_REGISTER(Filter, FilterReturn_Filter_status_Skip_, "Skip,skip");

// using FilterReturn_Filter_status_run_ = FilterReturn<Filter::status::Run>;
// IPIPE_REGISTER(Filter, FilterReturn_Filter_status_run_, "Run");

using FilterReturn_Filter_status_Run_ = FilterReturn<Filter::status::Run>;
IPIPE_REGISTER(Filter, FilterReturn_Filter_status_Run_, "Run,Continue,run");

// using FilterReturn_Filter_status_stop_ = FilterReturn<Filter::status::Break>;
// IPIPE_REGISTER(Filter, FilterReturn_Filter_status_stop_, "Break");

using FilterReturn_Filter_status_Stop_ = FilterReturn<Filter::status::Break>;
IPIPE_REGISTER(Filter, FilterReturn_Filter_status_Stop_, "Break,break");

using FilterReturn_Filter_status_SerialSkip_ = FilterReturn<Filter::status::SerialSkip>;
IPIPE_REGISTER(Filter, FilterReturn_Filter_status_SerialSkip_, "SerialSkip,serial_skip,serialskip");

using FilterReturn_Filter_status_GraphSkip_ = FilterReturn<Filter::status::SubGraphSkip>;
IPIPE_REGISTER(Filter, FilterReturn_Filter_status_GraphSkip_,
               "SubGraphSkip,subgraph_skip,subgraphskip");

using FilterReturn_Filter_status_Error_ = FilterReturn<Filter::status::Error>;
IPIPE_REGISTER(Filter, FilterReturn_Filter_status_Error_, "Error,error");

class FilterTarget : public Filter {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config,
                    dict /*dict_config*/) {
    params_ = std::unique_ptr<Params>(new Params({}, {"target"}, {}, {"target"}));
    if (!params_->init(config)) return false;
    target_ = params_->at("target");
    return true;
  };

  virtual status forward(dict input) {
    params_->check_and_update(input);
    if (params_->at("target") != target_) {
      return status::Skip;
    }
    auto iter = input->find(TASK_RESULT_KEY);
    if (iter == input->end()) return status::Break;
    (*input)[TASK_DATA_KEY] = iter->second;

    return status::Run;
  }

 private:
  std::unique_ptr<Params> params_;
  std::string target_;
};

IPIPE_REGISTER(Filter, FilterTarget, "Target")

class FilterHalfChance : public Filter {
 public:
  status forward(dict input) {
    if (rand() % 2 == 0)
      return status::Run;
    else
      return status::Skip;
  }
};

IPIPE_REGISTER(Filter, FilterHalfChance, "half_chance")

class result2key_1 : public Filter {
 public:
  status forward(dict input) {
    auto iter = input->find(TASK_RESULT_KEY);
    if (iter == input->end()) {
      // there must be sth wrong

      return status::Break;
    }
    (*input)[TASK_DATA_KEY] = iter->second;
    // input->erase(TASK_RESULT_KEY);
    (*input)["key_1"] = iter->second;
    return status::Run;
  }
};

IPIPE_REGISTER(Filter, result2key_1, "result2key_1")

class result2other : public Filter {
 public:
  status forward(dict input) {
    auto iter = input->find(TASK_RESULT_KEY);
    if (iter == input->end()) {
      // there must be sth wrong

      return status::Break;
    }
    (*input)[TASK_DATA_KEY] = iter->second;
    // input->erase(TASK_RESULT_KEY);
    (*input)["other"] = iter->second;
    return status::Run;
  }
};

IPIPE_REGISTER(Filter, result2other, "result2other")

}  // namespace ipipe
