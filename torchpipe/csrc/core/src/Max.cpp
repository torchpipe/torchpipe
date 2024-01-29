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
namespace ipipe {

class Max : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
    params_ = std::unique_ptr<Params>(
        new Params({{"Max::backend", "Identity"}, {"max", ""}}, {}, {}, {}));

    if (!params_->init(config)) return false;
    if (!params_->at("max").empty()) max_ = std::stoi(params_->at("max"));
    IPIPE_ASSERT(max_ > 0);
    backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("Max::backend")));
    if (backend_ && backend_->init(config, dict_config)) {
      backend_max_ = backend_->max();
      IPIPE_ASSERT(backend_max_ > 0);

      if (backend_->min() != 1) {
        SPDLOG_ERROR("backend_->min() != 1  min = {}", backend_->min());
        return false;
      }
      return true;
    }

    return false;
  };

  void forward(const std::vector<dict>& input_dicts) override {
    if (input_dicts.size() <= backend_max_) {
      backend_->forward(input_dicts);
    } else {
      const std::size_t batch_num = input_dicts.size() / backend_max_;
      for (std::size_t i = 0; i < batch_num; ++i) {
        backend_->forward(std::vector<dict>(input_dicts.begin() + i * backend_max_,
                                            input_dicts.begin() + (i + 1) * backend_max_));
      }
      int left_num = input_dicts.size() - batch_num * backend_max_;
      assert(left_num == 0 || left_num >= backend_->min());
      if (left_num > 0)
        backend_->forward(
            std::vector<dict>(input_dicts.begin() + batch_num * backend_max_, input_dicts.end()));
    }
  }
  /// @brief  UINT32_MAX
  virtual uint32_t max() const override final { return max_; };
  /// 等于子后端对应值。
  virtual uint32_t min() const override final { return backend_->min(); };

 private:
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<Params> params_;
  uint32_t max_{UINT32_MAX};
  uint32_t backend_max_{0};
};

IPIPE_REGISTER(Backend, Max, "Max");
}  // namespace ipipe