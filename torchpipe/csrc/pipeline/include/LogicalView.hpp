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
namespace ipipe {

class LogicalView : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict) override;

  void forward(const std::vector<dict>& inputs) override;

  uint32_t max() const override {
    return UINT32_MAX;  // UINT_MAX
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~LogicalView() = default;
#endif

 private:
  struct LogicalCommand {
    enum class MapType { map, reduce, normal };
    std::string filter_str;
    std::set<std::string> next;
    std::set<std::string> previous;
    std::unordered_map<std::string, std::pair<std::string, MapType>>
        map_reduce;                               // previous.size()>1
    std::pair<std::string, MapType> map_reduce2;  // previous.size()<=1 or can achieve from anyone

    std::set<std::string> sync;
    std::string physical_name;

    Filter filter;
  };

  using LogicalCommands = std::unordered_map<std::string /**logical node name*/, LogicalCommand>;
  LogicalCommands logical_orders_;
  LogicalCommands get_orders(const std::unordered_map<std::string, std::string>& config);

  bool init(mapmap config);
};

}  // namespace ipipe