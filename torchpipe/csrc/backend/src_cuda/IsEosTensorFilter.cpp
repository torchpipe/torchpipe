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

#include <torch/torch.h>
#include <fstream>
#include "base_logging.hpp"
#include "reflect.h"
#include "dict.hpp"
#include "params.hpp"
#include "threadsafe_kv_storage.hpp"
#include "filter.hpp"

namespace ipipe {

class IsRequestEosFilter : public SingleBackend {
 public:
  virtual void forward(dict input_dict) override {
    static auto& storage = ThreadSafeKVStorage::getInstance();

    auto& input = *input_dict;

    auto iter = input_dict->find("request_id");
    IPIPE_ASSERT(iter != input_dict->end());
    std::string request_id = any_cast<std::string>(iter->second);
    auto& storage_kv = storage.get(request_id);

    if (storage_kv.get("is_eos")) {
      SPDLOG_DEBUG("IsRequestEosFilter: is_eos is true");
      input["filter"] = Filter::status::Run;
    } else
      input["filter"] = Filter::status::Skip;
  }
};

IPIPE_REGISTER(Backend, IsRequestEosFilter, "IsRequestEosFilter");

}  // namespace ipipe