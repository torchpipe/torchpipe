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

#pragma once

#include <memory>
#include <set>
#include <unordered_map>

#include "Backend.hpp"
#include "dict.hpp"
#include "config_parser.hpp"
#include "event.hpp"
#include "filter.hpp"
#include "graph.hpp"
#include "Schedule.hpp"
#include "threadsafe_queue.hpp"
#include "EventBackend.hpp"
namespace ipipe {

/**
 * @brief 多节点调度后端。支持：
 * - DAG图
 * - 子图独立调度
 * - filter
 * - map
 */
class PhysicalView : public SingleEventBackend {
 public:
  /**
   * @brief 初始化 WIP
   *
   */
  bool init(const std::unordered_map<std::string, std::string>& config, dict) override;

  void forward(dict input, std::shared_ptr<SimpleEvents> event,
               std::string node_name) noexcept override;

  /**
   *
   * @return UINT32_MAX
   */
  // uint32_t max() const override {
  //   return 1;  // UINT_MAX
  // }

 private:
  // std::unordered_map<std::string, std::string> golbal_settings_;

  bool init(mapmap config);

  std::unordered_map<std::string, std::unique_ptr<Backend>> backends_;
  std::unordered_map<std::string, std::string> logical2physical_;
  std::unordered_map<std::string, std::string> borrow_from_;
  dict dict_config_;
  // std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config_;
};

}  // namespace ipipe