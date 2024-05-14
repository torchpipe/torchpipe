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

#include "torchpipe/extension.h"
#include <torch/torch.h>
namespace ipipe {
class PY2CPP : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto data = *input_dicts[0];

    std::string str = any_cast<std::string>(data["str"]);
    std::string bytes = any_cast<std::string>(data["bytes"]);
    torch::Tensor Tensor = any_cast<torch::Tensor>(data["torch.Tensor"]);
    bool bool_value = any_cast<bool>(data["bool"]);
    float double_value = any_cast<float>(data["float"]);
    int value = any_cast<int>(data["int"]);
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(Backend, PY2CPP, "PY2CPP");

class ListPY2CPP : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto& data = *input_dicts[0];
    if (data["str"].size() == 0) {
      IPIPE_ASSERT(data["str"].type() == typeid(std::vector<void*>));
    }

    std::vector<std::string> str = any_cast<std::vector<std::string>>(data["str"]);
    std::vector<std::string> bytes = any_cast<std::vector<std::string>>(data["bytes"]);
    std::vector<torch::Tensor> Tensor = any_cast<std::vector<torch::Tensor>>(data["torch.Tensor"]);
    std::vector<bool> bool_value = any_cast<std::vector<bool>>(data["bool"]);
    std::vector<float> double_value = any_cast<std::vector<float>>(data["float"]);
    std::vector<int> value = any_cast<std::vector<int>>(data["int"]);
    if (str.empty()) IPIPE_ASSERT(data["str"].type() == typeid(std::vector<std::string>));
    data.clear();
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(Backend, ListPY2CPP, "ListPY2CPP");

class SetPY2CPP : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto& data = *input_dicts[0];
    if (data["str"].size() == 0) {
      IPIPE_ASSERT(data["str"].type() == typeid(std::unordered_set<void*>));
      IPIPE_ASSERT(data["str"].type() == typeid(std::set<void*>));
      IPIPE_ASSERT(data["str"].type() == any(UnknownContainerTag()).type());
    }

    std::unordered_set<std::string> str = any_cast<std::unordered_set<std::string>>(data["str"]);

    std::unordered_set<std::string> bytes =
        any_cast<std::unordered_set<std::string>>(data["bytes"]);
    // std::unordered_set<torch::Tensor> Tensor =
    //     any_cast<std::unordered_set<torch::Tensor>>(data["torch.Tensor"]);
    std::unordered_set<bool> bool_value = any_cast<std::unordered_set<bool>>(data["bool"]);
    std::unordered_set<float> double_value = any_cast<std::unordered_set<float>>(data["float"]);
    const std::unordered_set<int>& value = any_cast<std::unordered_set<int>>(data["int"]);
    if (str.empty()) {
      IPIPE_ASSERT(data["str"].type() == typeid(std::unordered_set<std::string>));
      IPIPE_ASSERT(data["str"].type() != typeid(std::unordered_set<void*>));
    }
    data.clear();
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(Backend, SetPY2CPP, "SetPY2CPP");

template <class T>
using StrMap = std::unordered_map<std::string, T>;

class StrMapPY2CPP : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto& data = *input_dicts[0];
    if (data["str"].size() == 0) {
      IPIPE_ASSERT(data["str"].type() == typeid(std::unordered_set<void*>));
      IPIPE_ASSERT(data["str"].type() == typeid(std::set<void*>));
      IPIPE_ASSERT(data["str"].type() == any(UnknownContainerTag()).type());
    }

    StrMap<std::string> str = any_cast<StrMap<std::string>>(data["str"]);

    StrMap<std::string> bytes = any_cast<StrMap<std::string>>(data["bytes"]);
    StrMap<torch::Tensor> Tensor = any_cast<StrMap<torch::Tensor>>(data["torch.Tensor"]);
    StrMap<bool> bool_value = any_cast<StrMap<bool>>(data["bool"]);
    StrMap<float> double_value = any_cast<StrMap<float>>(data["float"]);
    const StrMap<int>& value = any_cast<StrMap<int>>(data["int"]);
    if (str.empty()) {
      IPIPE_ASSERT(data["str"].type() == typeid(StrMap<std::string>));
      IPIPE_ASSERT(data["str"].type() != typeid(std::unordered_set<void*>));
    }
    data.clear();
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(Backend, StrMapPY2CPP, "StrMapPY2CPP");
}  // namespace ipipe