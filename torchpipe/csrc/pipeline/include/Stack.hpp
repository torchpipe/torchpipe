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
#include "LogicalGraph.hpp"
#include <algorithm>
#include "base_logging.hpp"
// #include "Schedule.hpp"
#include "threadsafe_queue.hpp"
namespace ipipe {
// only one thread can visit Stack at once.
struct Stack {
  // 新增独立节点会折叠到next_config.keys()中唯一父节点；注意：此操作很危险。
  void update_graph(const std::unordered_map<std::string, std::set<std::string>>& next_config,
                    const std::set<std::string>& del_nodes) {
    // if (next_config.empty()) {
    //   assert(false);
    //   throw std::runtime_error("Empty config is not reported.");
    // }
    std::unordered_map<std::string, std::string> fold_map;
    // auto old_graph = graph;
    graph = graph->update(next_config, del_nodes, fold_map);
    if (graph == nullptr) {
      // 需要恢复graph 和fold_map， 太麻烦；
      throw std::runtime_error("graph update failed.");
    }

    // update_fold(fold_map);
    update_status();  // 以他为跟节点重建子图

    return;
  }
  void serial_skip(const std::string& node_name) {
    std::set<std::string> sub_nexts;
    if (!graph->get_serial_nexts(node_name, sub_nexts))
      throw std::runtime_error("serial_skip failed.");

    update_graph({{node_name, sub_nexts}}, {});
  }

  void stop(const std::string& node_name) {
    const auto& ne = graph->get_all();
    start_node = node_name;
    // stop_node = node_name;
    // update_graph({{node_name, {}}}, {});
    update_graph({}, ne);
  }

  void graph_skip(const std::string& node_name) {
    std::set<std::string> sub_nexts;
    if (!graph->get_subgraph_nexts(node_name, sub_nexts))
      throw std::runtime_error("graph_skip failed.");

    update_graph({{node_name, sub_nexts}}, {});
  }
  bool valid(const std::string& node) {
    return graph && graph->get_subtree(start_node).count(node) != 0;
  }
  void update_status() {
    waiting_nodes = graph->get_subtree(start_node);
    this->num_nodes = this->waiting_nodes.size();
    // this->waiting_nodes.erase(root);

    for (auto iter_processed = processed.begin(); iter_processed != processed.end();) {
      if (waiting_nodes.count(iter_processed->first) == 0) {
        iter_processed = processed.erase(iter_processed);
      } else {
        ++iter_processed;
      }
    }

    for (auto iter = waiting_nodes.begin(); iter != waiting_nodes.end();) {
      auto iter_processed = non_waiting_nodes.find(*iter);
      if (iter_processed != non_waiting_nodes.end()) {
        iter = waiting_nodes.erase(iter);
        // this->num_nodes--;
      } else {
        ++iter;
      }
    }
    update_end_nodes();
  }

  void update_end_nodes() {
    end_nodes.clear();
    const auto& subs = graph->get_subtree(start_node);
    for (const auto& item : subs) {
      if (graph->get_next(item).empty()) {
        end_nodes.insert(item);
      }
    }
    IPIPE_ASSERT(!end_nodes.empty());
  }

  friend class PipelineV3;
  // friend class LogicalGraph;
  void set_filter_status(std::string name, Filter::status stat) { filter_status[name] = stat; }

 private:
  // 此处禁止异常，否则Pipeline可能无法停止
  // void update_fold(const std::unordered_map<std::string, std::string>& fold_map) noexcept {
  //   for (const auto& item : fold_map) {
  //     waiting_nodes.erase(item.first);
  //     // processed[item.first] = nullptr;
  //     // processed[item.first] = processed.at(item.second);
  //   }
  // };

  void update_processed(std::string node_name, dict data) noexcept {
    processed[node_name] = data;
    // const std::unordered_map<std::string, std::string>& fold_map = graph->get_fold_map();
    // for (const auto& item : fold_map) {
    //   if (item.second == node_name) {
    //     processed[item.first] = data;
    //   }
    // }
  }
  dict get_mapped_previous(std::string node_name) {
    return graph->get_mapped_previous(node_name, processed);
  }
  // std::string root;
  std::unordered_map<std::string, dict> processed;
  std::unordered_map<std::string, Filter::status> filter_status;
  // std::unordered_map<std::string, Filter::status> filter_status;
  std::set<std::string> waiting_nodes;
  std::set<std::string> non_waiting_nodes;

  dict input_data;
  std::shared_ptr<SimpleEvents> input_event;
  std::exception_ptr exception;
  uint32_t num_nodes;
  std::string start_node;
  std::set<std::string> end_nodes;
  // std::string stop_node;

  std::size_t task_queue_index = 0;
  bool need_update_processed{false};

  bool allStopped() { return waiting_nodes.size() + processed.size() == num_nodes; }
  bool allFinished() {
    for (const auto& item : end_nodes) {
      if (processed.find(item) == processed.end()) return false;
    };
    return true;
  }
  void clear() {
    processed.clear();
    waiting_nodes.clear();
    input_data = nullptr;
    input_event = nullptr;
    sub_stacks.clear();
    graph = nullptr;
    // alive = false;
  }
  // bool alive{true};
  std::shared_ptr<LogicalGraph> graph;

  Stack* parent{nullptr};
  std::vector<std::shared_ptr<Stack>> sub_stacks;
};

}  // namespace ipipe