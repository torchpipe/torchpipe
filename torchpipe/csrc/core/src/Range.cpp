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

#include "params.hpp"

#include "Backend.hpp"
#include "reflect.h"
#include "base_logging.hpp"
#include "exception.hpp"
#include "exception.hpp"
#include "ipipe_common.hpp"
#include <numeric>
namespace ipipe {

// Use dynamic programming to determine whether the integer 'target' can be represented as the sum
// of multiple numbers within the [min_value, max_value] range, and provide a result with the
// largest possible values. for example, canBeRepresented(2,9,10) return true, and result == {8,2}
bool canBeRepresented(uint32_t min_value, uint32_t max_value, uint32_t target,
                      std::vector<uint32_t>& result) {
  // note max_value>=min_value>=1, target>=1
  if (target < min_value)
    return false;
  else if (target <= max_value) {
    result.push_back(target);
    return true;
  }
  if (min_value == 1) {  // optinal
    while (target >= max_value) {
      result.push_back(max_value);
      target -= max_value;
    }
    if (target != 0) result.push_back(target);
    return true;
  }

  std::vector<int> dp(target + 1, -1);
  dp[0] = 0;

  for (uint32_t i = min_value; i <= max_value; ++i) {
    for (uint32_t j = i; j <= target; ++j) {
      if (dp[j - i] != -1) {
        dp[j] = i;
      }
    }
  }

  if (dp[target] == -1) {
    return false;
  }

  for (int i = target; i > 0; i -= dp[i]) {
    result.push_back(dp[i]);
  }

  return true;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 *
 */
class Range : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
    params_ =
        std::unique_ptr<Params>(new Params({{"Range::backend", "Identity"}}, {"range"}, {}, {}));

    if (!params_->init(config)) return false;

    std::vector<std::string> range_minmax;
    TRACE_EXCEPTION(range_minmax = str_split(params_->at("range"), ',', false));
    IPIPE_CHECK(range_minmax.size() == 2, "range should be like 'min,max'");
    TRACE_EXCEPTION(min_ = std::stoi(range_minmax[0]));
    TRACE_EXCEPTION(max_ = std::stoi(range_minmax[1]));

    backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("Range::backend")));
    if (backend_ && backend_->init(config, dict_config)) {
      if (backend_->min() > max_) {
        SPDLOG_ERROR("backend: min = {} max = {}. illegal target range: [{} {}]", backend_->min(),
                     backend_->max(), min_, max_);
        return false;
      }
      for (std::size_t i = min_; i <= max_; ++i) {
        std::vector<uint32_t> result;
        if (!canBeRepresented(backend_->min(), backend_->max(), i, result)) {
          SPDLOG_ERROR("backend: min = {} max = {}. illegal target range: [{} {}]", backend_->min(),
                       backend_->max(), min_, max_);
          return false;
        }
        represent_map_[i] = result;
        for (const auto& j : result) {
          SPDLOG_DEBUG("target: {} -> backend: {}", i, j);
        }
        IPIPE_ASSERT(i == std::accumulate(result.begin(), result.end(), 0));
      }
      SPDLOG_DEBUG("backend: min = {} max = {}.  target range: [{} {}]", backend_->min(),
                   backend_->max(), min_, max_);
      return true;
    }

    return false;
  };

  void forward(const std::vector<dict>& input_dicts) override {
    if (input_dicts.size() <= max_) {
      backend_->forward(input_dicts);
    } else {
      auto represented = represent_map_[input_dicts.size()];

      uint32_t passed{0};
      for (std::size_t i = 0; i < represented.size(); ++i) {
        backend_->forward(std::vector<dict>(input_dicts.begin() + passed,
                                            input_dicts.begin() + passed + represented[i]));
        passed += represented[i];
      }
    }
  }
  virtual uint32_t max() const override final { return max_; };
  virtual uint32_t min() const override final { return min_; };

 private:
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<Params> params_;
  uint32_t max_{1};
  uint32_t min_{1};
  std::unordered_map<uint32_t, std::vector<uint32_t>> represent_map_;
};

IPIPE_REGISTER(Backend, Range, "Range");
#endif
}  // namespace ipipe