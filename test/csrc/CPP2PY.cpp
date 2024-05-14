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
#include "opencv2/core.hpp"
// std::unordered_map<std::string> data_map

namespace ipipe {
struct empty_any_struct {};
std::unordered_map<std::string, any> get_data_map() {
  std::unordered_map<std::string, any> data;

  torch::Tensor Tensor;
  data["std::string"] = std::string();
  data["cv::Mat"] = cv::Mat(1, 2, CV_8UC3);
  data["torch::Tensor"] = Tensor;
  data["bool"] = true;
  data["float"] = float(1.23);
  data["double"] = double(1.23);
  data["int"] = 123;
  data["unsigned int"] = (unsigned int)123;
  data["char"] = char(1);
  data["unsigned char"] = (unsigned char)1;
  // data["any"] = empty_any_struct();
  return data;
}

template <class TT>
using vector2 = std::vector<std::vector<TT>>;

template <template <class... Args> class container = std::vector>
std::unordered_map<std::string, any> get_vector_map() {
  std::unordered_map<std::string, any> data;

  data["std::string"] = container<std::string>{std::string()};
  data["cv::Mat"] = container<cv::Mat>{cv::Mat(1, 2, CV_8UC3)};
  data["torch::Tensor"] = container<torch::Tensor>{torch::empty({6})};
  // data["bool"] = container<bool>{true};
  data["float"] = container<float>{{float(1.23)}};
  data["double"] = container<double>{{double(1.23)}};
  data["int"] = container<int>{{123}};
  data["unsigned int"] = container<unsigned int>{{(unsigned int)123}};
  data["char"] = container<char>{{char(1)}};
  data["unsigned char"] = container<unsigned char>{{(unsigned char)1}};
  std::vector<any> data_any = {any(1), any(std::vector<torch::Tensor>({torch::empty({6})}))};
  data["any"] = data_any;
  std::unordered_map<std::string, any> data_any_str_dict{
      {"int", any(1)}, {"torch::Tensor", any(std::vector<torch::Tensor>({torch::empty({6})}))}};
  data["any_str_dict"] = data_any_str_dict;
  return data;
}

template <template <class... Args> class container = vector2>
std::unordered_map<std::string, any> get_vector2_map() {
  std::unordered_map<std::string, any> data;

  // data["std::string"] = container<std::string>{std::string()};
  // data["cv::Mat"] = container<cv::Mat>{cv::Mat(1, 2, CV_8UC3)};
  // data["torch::Tensor"] = container<torch::Tensor>{torch::empty({6})};
  // data["bool"] = container<bool>{true};
  data["float"] = container<float>{{float(1.23)}};
  data["double"] = container<double>{{double(1.23)}};
  data["int"] = container<int>{{123}};
  data["unsigned int"] = container<unsigned int>{{(unsigned int)123}};
  data["char"] = container<char>{{char(1)}};
  data["unsigned char"] = container<unsigned char>{{(unsigned char)1}};
  data["empty"] = container<unsigned char>();
  return data;
}

std::unordered_map<std::string, ipipe::any> get_set_map() {
  std::unordered_map<std::string, ipipe::any> data;

  data["std::string"] = std::unordered_set<std::string>{std::string()};
  // data["cv::Mat"] = std::unordered_set<cv::Mat>{cv::Mat(1, 2, CV_8UC3)};
  // data["torch::Tensor"] = std::unordered_set<torch::Tensor>{torch::empty({6})};
  data["bool"] = std::unordered_set<bool>{true};
  data["float"] = std::unordered_set<float>{float(1.23)};
  data["double"] = std::unordered_set<double>{double(1.23)};
  data["int"] = std::unordered_set<int>{123};
  data["unsigned int"] = std::unordered_set<unsigned int>{(unsigned int)123};
  data["char"] = std::unordered_set<char>{char(1)};
  data["unsigned char"] = std::unordered_set<unsigned char>{(unsigned char)1};
  return data;
}

template <class T>
std::unordered_map<std::string, T> init_str_dict(T data) {
  return std::unordered_map<std::string, T>{{"", data}};
}
std::unordered_map<std::string, any> get_str_dict_map() {
  std::unordered_map<std::string, any> data;

  data["std::string"] = init_str_dict(std::string());
  data["cv::Mat"] = init_str_dict(cv::Mat(1, 2, CV_8UC3));
  data["torch::Tensor"] = init_str_dict(torch::empty({6}));
  data["bool"] = init_str_dict(true);
  data["float"] = init_str_dict(float(1.23));
  data["double"] = init_str_dict(double(1.23));
  data["int"] = init_str_dict(123);
  data["unsigned int"] = init_str_dict((unsigned int)123);
  data["char"] = init_str_dict(char(1));
  data["unsigned char"] = init_str_dict((unsigned char)1);

  std::vector<any> data_any = {any(1), any(std::vector<torch::Tensor>({torch::empty({6})}))};
  data["any"] = data_any;
  std::unordered_map<std::string, any> data_any_str_dict = {
      {"int", any(1)},
      {"torch::Tensor", any(std::vector<torch::Tensor>({torch::empty({6}), torch::empty({6})}))}};
  data["any_str_dict"] = data_any_str_dict;
  return data;
}

class CPP2PY : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto& data = *input_dicts[0];

    data = get_data_map();
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(Backend, CPP2PY, "CPP2PY");

class ListCPP2PY : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto& data = *input_dicts[0];
    data = get_vector_map<>();
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(Backend, ListCPP2PY, "ListCPP2PY");

class List2CPP2PY : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto& data = *input_dicts[0];
    data = get_vector2_map();
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(Backend, List2CPP2PY, "List2CPP2PY");

class SetCPP2PY : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto& data = *input_dicts[0];
    data = get_set_map();
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(Backend, SetCPP2PY, "SetCPP2PY");

template <class T>
using StrMap = std::unordered_map<std::string, T>;

}  // namespace ipipe

class StrMapCPP2PY : public ipipe::Backend {
 public:
  void forward(const std::vector<ipipe::dict>& input_dicts) override {
    // (*input_dicts[0])["result"] = input_dicts[0]->at("data");
    auto& data = *input_dicts[0];
    data = ipipe::get_str_dict_map();
  }
  uint32_t max() const override final { return 1; };
};

IPIPE_REGISTER(ipipe::Backend, StrMapCPP2PY, "StrMapCPP2PY");