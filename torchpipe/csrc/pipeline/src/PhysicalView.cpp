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

// #include "node.hpp"

#include "PhysicalView.hpp"
#include <functional>

#include <cassert>
#include <string>
#include <vector>

#include "config_parser.hpp"

#include <algorithm>
#include <cstdlib>
#include <future>
#include "base_logging.hpp"
#include "event.hpp"
#include "filter.hpp"
#include "reflect.h"
#include "dict_helper.hpp"

#include "BaselineSchedule.hpp"
namespace ipipe {

bool PhysicalView::init(const std::unordered_map<std::string, std::string>& config,
                        dict dict_config) {
  dict_config_ = dict_config;
  mapmap global_config;
  if (dict_config && dict_config->find("config") != dict_config->end()) {
    global_config = any_cast<mapmap>((*dict_config)["config"]);
    return init(global_config);
  } else if (!config.empty()) {
    global_config[TASK_DEFAULT_NAME_KEY] = config;
    return init(global_config);
  }
  return false;
}

bool PhysicalView::init(mapmap config) {
  if (config.size() == 1 && config.find("global") != config.end()) {
    config[TASK_DEFAULT_NAME_KEY] = config["global"];
  }
  config.erase("global");

  for (auto iter = config.begin(); iter != config.end(); ++iter) {
    auto iter_dot = iter->first.find(".");
    if (iter_dot != std::string::npos) {
      logical2physical_.emplace(iter->first, iter->first.substr(0, iter_dot));
      continue;
    }

    auto iter_pipeline_backend = iter->second.find("PhysicalView::backend");
    if (iter_pipeline_backend == iter->second.end()) {
      backends_[iter->first] = std::make_unique<BaselineSchedule>();
    } else {
      std::string tmp_name = any_cast<std::string>(iter_pipeline_backend->second);
      backends_[iter->first] = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, tmp_name));
    }
    SPDLOG_INFO("start init node: {}", iter->first);
    if (!backends_[iter->first] || !backends_[iter->first]->init(iter->second, dict_config_)) {
      SPDLOG_ERROR("{}: init failed", iter->first);
      backends_.clear();
      return false;
    }
  }

  return true;
}

class RAII_FUNCTION {
 public:
  RAII_FUNCTION(std::function<void()> func) : func_(func) {}
  ~RAII_FUNCTION() {
    if (func_) func_();
  }

 private:
  std::function<void()> func_;
};

void PhysicalView::forward(dict input, std::shared_ptr<SimpleEvents> event,
                           std::string node_name) noexcept {
  if (node_name.empty()) {
    if (backends_.size() == 1) {
      node_name = backends_.begin()->first;
    }
  } else if (node_name.find('.') != std::string::npos) {
    auto iter_ = logical2physical_.find(node_name);
    if (iter_ != logical2physical_.end()) {
      node_name = iter_->second;
    } else {
      event->set_exception_and_notify_all(
          std::make_exception_ptr(std::runtime_error("can not find `" + node_name + "`")));
      return;
    }
  }

  auto iter_backend = backends_.find(node_name);
  if (iter_backend == backends_.end()) {
    SPDLOG_ERROR("found no node_name: `{}`.", node_name);
    event->set_exception_and_notify_all(
        std::make_exception_ptr(std::runtime_error("node_name=" + node_name)));
    return;
  }

  iter_backend->second->forward({input});
}

IPIPE_REGISTER(Backend, PhysicalView, "PhysicalView");

}  // namespace ipipe
