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

#include "torchpipe/extension.h"

class CustomIdentity : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    (*input_dicts[0])["result"] = input_dicts[0]->at("data");
  }
  uint32_t max() const override final { return 1; };
};

namespace ipipe {
IPIPE_REGISTER(Backend, CustomIdentity, "CustomIdentity");
}  // namespace ipipe