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

#include <set>
#include <unordered_map>
#include "LogicalGraph.hpp"
#include <algorithm>
#include "base_logging.hpp"
#include "params.hpp"

namespace ipipe {

std::string LogicalGraph::get_default_filter(const std::string& node_name) const {
  auto iter = node_configs_.find(node_name);
  if (iter == node_configs_.end()) {
    SPDLOG_ERROR("{}: no config", node_name);
    throw std::runtime_error(node_name + ":  no config");
  }
  if (iter->second.indegree == 1 && iter->second.map_reduce.empty() &&
      iter->second.map_reduces.empty()) {
    static auto tmp = []() {
      SPDLOG_WARN(
          "Using the `swap` filter (default for non-root node)."
          " This will cause a 'Break' status if no TASK_RESULT_KEY was found. Please "
          "ensure you are using the correct filter.");
      return 0;
    }();

    return "swap";
  } else {
    return "Run";
  }
}
std::set<std::string> LogicalGraph::get_all_previous(std::string node) {
  std::set<std::string> all_previous;
  for (auto& item : node_configs_) {
    if (sub_tree_[item.first].count(node) == 1) {
      all_previous.insert(item.first);
    }
  }
  all_previous.erase(node);
  return all_previous;
}
std::set<std::string> LogicalGraph::extract_dependent_roots_from_next() {
  std::set<std::string> roots;
  for (auto& item : node_configs_) {
    item.second.indegree = 0;
  }

  for (const auto& item : node_configs_) {
    for (const auto& next : item.second.next) {
      node_configs_[next].indegree++;
    }
  }
  for (const auto& item : node_configs_) {
    if (item.second.indegree == 0 && !item.second.next.empty()) {
      roots.insert(item.first);
    }
  }
  return roots;
}

bool LogicalGraph::finalize() {
  sortted_.clear();
  roots_.clear();
  sub_tree_.clear();
  for (auto& item : node_configs_) {
    item.second.previous.clear();
    item.second.indegree = 0;
  }

  for (const auto& item : node_configs_) {
    for (const auto& next : item.second.next) {
      node_configs_[next].indegree++;
      node_configs_[next].previous.insert(item.first);
    }
  }
  // if (!check_map()) return false;

  std::queue<std::string> zero_indegree_queue;
  auto node_configs = node_configs_;
  for (const auto& item : node_configs) {
    if (item.second.indegree == 0) {
      zero_indegree_queue.push(item.first);
      if (std::find(sortted_.begin(), sortted_.end(), item.first) == sortted_.end())
        sortted_.push_back(item.first);
      roots_.insert(item.first);
    }
  }

  int count = 0;  // 计数，记录当前已经输出的顶点数
  while (!zero_indegree_queue.empty()) {
    auto v = zero_indegree_queue.front();  // 从队列中取出一个顶点
    zero_indegree_queue.pop();

    ++count;
    for (auto beg = node_configs[v].next.begin(); beg != node_configs[v].next.end(); ++beg)
      if (!(--node_configs[*beg].indegree)) {
        zero_indegree_queue.push(*beg);  // 若入度为0，则入栈
        if (std::find(sortted_.begin(), sortted_.end(), *beg) == sortted_.end())
          sortted_.push_back(*beg);
      }
  }

  if (count < node_configs.size()) {
    sortted_.clear();
    roots_.clear();
    return false;  // 没有输出全部顶点，有向图中有回路
  } else {
    return split();  // 拓扑排序成功
  }
}
bool LogicalGraph::split() {
  for (const auto& item : node_configs_) {
    const auto& r = item.first;
    // stack_name_[r] = r;
    const auto& config = item.second;
    std::set<std::string> waiting_nodes;
    waiting_nodes.insert(r);
    auto nexts = config.next;
    while (true) {
      if (nexts.empty()) break;
      waiting_nodes.insert(nexts.begin(), nexts.end());
      std::set<std::string> nexts_nexts;
      for (auto data : nexts) {
        nexts_nexts.insert(node_configs_[data].next.begin(), node_configs_[data].next.end());
      }
      std::swap(nexts_nexts, nexts);
    }
    sub_tree_[r] = waiting_nodes;
  }

  // 检查一个联通分支有且只有一个root
  for (const auto& item : node_configs_) {
    const auto& r = item.first;
    std::size_t num = 0;
    for (const auto& root : roots_) {
      if (sub_tree_[root].count(r) == 1) num++;
    }
    if (num != 1) {
      SPDLOG_ERROR("`" + r + "` node has multiple roots.");
      return false;
    }
  }

  return true;
}

LogicalGraph& LogicalGraph::set_map_reduce(const std::string& v, const std::string& config,
                                           bool is_reduce) {
  if (node_configs_.find(v) == node_configs_.end()) {
    node_configs_[v] = LogicalNodeConfig();
  }
  auto generated_data = generate_map(config);
  const auto& previous = get_previous(v);
  if (generated_data.size() == 1 && (generated_data.begin()->first.empty())) {
    const auto& items = generated_data.begin()->second;
    for (const auto& item : items) {
      // auto type = MapReduceType::normal;
      // if (item.second.map_type != MapInfo::MapType::replace) {
      //   type = is_reduce ? MapReduceType::reduce : MapReduceType::map;
      //   node_configs_[v].map_reduce_type = type;
      // }
      node_configs_[v].map_reduce[item.first] =
          item.second.value;  // MapSrc(item.second.value, type);
    }
  } else {
    for (auto iter = generated_data.begin(); iter != generated_data.end(); ++iter) {
      const auto& items = iter->second;
      auto config_map = std::unordered_map<std::string, LogicalGraph::MapSrc>();
      for (const auto& item : items) {
        // auto type = MapReduceType::normal;
        // if (item.second.map_type != MapInfo::MapType::replace) {
        //   type = is_reduce ? MapReduceType::reduce : MapReduceType::map;
        //   node_configs_[v].map_reduce_type = type;
        // };
        config_map[item.first] = item.second.value;  // MapSrc(item.second.value, type);
      }
      node_configs_[v].map_reduces[iter->first] = config_map;
    }
  }

  return *this;
}

bool LogicalGraph::check_map() {
  for (const auto& item : node_configs_) {
    if (item.second.previous.size() > 1 && item.second.map_reduces.empty() &&
        item.second.map_reduce.empty()) {
      SPDLOG_ERROR("`" + item.first +
                   "` has multiple previous nodes, however no `map` configuration.");
      return false;
    }
  }

  std::unordered_map<std::string, std::string> old_new_pair;
  for (const auto& item : node_configs_) {
    auto all_previous = get_all_previous(item.first);
    for (const auto& map_from : item.second.map_reduces) {
      if (!all_previous.count(map_from.first)) {
        auto iter = fold_map_.find(map_from.first);
        if (iter == fold_map_.end()) {
          SPDLOG_ERROR("`" + item.first + "` was not a subnode of `" + map_from.first + "`.");
          return false;
        }
        if (!all_previous.count(iter->second)) {
          SPDLOG_ERROR("`" + iter->second + "`  `" + "`.");
          return false;
        }
        old_new_pair[map_from.first] = iter->second;
      }
    }
  }
  for (auto& item : node_configs_) {
    item.second.references = item.second.next;
    for (const auto& map_key : old_new_pair) {
      auto iter = item.second.map_reduces.find(map_key.first);
      if (iter != item.second.map_reduces.end()) {
        item.second.map_reduces[map_key.second] = iter->second;
        item.second.map_reduces.erase(iter);
      }
    }
  }

  for (auto& item : node_configs_) {
    auto all_previous = get_all_previous(item.first);
    for (const auto& map_from : item.second.map_reduces) {
      if (!all_previous.count(map_from.first)) return false;
      node_configs_[map_from.first].references.insert(item.first);
    }
  }
  return true;
}
};  // namespace ipipe
