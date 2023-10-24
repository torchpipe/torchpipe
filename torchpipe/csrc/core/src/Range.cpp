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
#include "exception.hpp"

namespace ipipe {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/**
 * @brief 未整理，请勿使用
 *
 */
class Range : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
    params_ = std::unique_ptr<Params>(
        new Params({{"min", "1"}, {"Range::backend", "Identity"}}, {"max"}, {}, {}));

    if (!params_->init(config)) return false;

    TRACE_EXCEPTION(min_ = std::stoi(params_->at("min")));
    TRACE_EXCEPTION(max_ = std::stoi(params_->at("max")));
    backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("Range::backend")));
    if (backend_ && backend_->init(config, dict_config)) {
      // max_ = backend_->max();
      if (backend_->min() <= min_ && backend_->max() >= max_) {
        return true;
      } else if (backend_->min() != 1) {
        SPDLOG_ERROR("backend: min = {} max = {}. illegal target range: [{} {}]", backend_->min(),
                     backend_->max(), min_, max_);
        return false;
      }
      return true;
    }

    return false;
  };

  void forward(const std::vector<dict>& input_dicts) override {
    if (input_dicts.size() <= max_) {
      backend_->forward(input_dicts);
    } else {
      const std::size_t batch_num = input_dicts.size() / max_;
      for (std::size_t i = 0; i < batch_num; ++i) {
        backend_->forward(std::vector<dict>(input_dicts.begin() + i * max_,
                                            input_dicts.begin() + (i + 1) * max_));
      }
      int left_num = input_dicts.size() - batch_num * max_;
      assert(left_num == 0 || left_num >= backend_->min());
      if (left_num > 0)
        backend_->forward(
            std::vector<dict>(input_dicts.begin() + batch_num * max_, input_dicts.end()));
    }
  }
  virtual uint32_t max() const override final { return max_; };
  virtual uint32_t min() const override final { return min_; };

 private:
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<Params> params_;
  uint32_t max_{1};
  uint32_t min_{1};
};

IPIPE_REGISTER(Backend, Range, "Range");
#endif
}  // namespace ipipe