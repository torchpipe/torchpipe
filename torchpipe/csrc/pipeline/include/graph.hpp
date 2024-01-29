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
#include <algorithm>
#include <set>

#include <set>
#include <stack>
#include <unordered_map>
#include <vector>
#include <stdexcept>
namespace ipipe {
template <typename T>
class Graph {
 public:
  //   Graph() = default;
  //   ~Graph() = default;
  void add(const T& a, const T& b) {
    auto iter = next_nodes_.find(a);
    if (iter != next_nodes_.end()) {
      iter->second.insert(b);
    } else {
      next_nodes_[a] = std::set<T>{b};
    }
    // add(b);
  }

  void add(const T& a, const std::vector<T>& b) {
    auto iter = next_nodes_.find(a);
    if (iter != next_nodes_.end()) {
      for (const auto& item : b) {
        iter->second.insert(item);
      }

    } else {
      next_nodes_[a] = std::set<std::string>(b.begin(), b.end());
    }
    // for (const auto& item : b) {
    //   add(item);
    // }
  }

  void add(const T& a) {
    auto iter = next_nodes_.find(a);
    if (iter == next_nodes_.end()) next_nodes_[a] = std::set<T>();
  }

  std::vector<T> sort() {
    if (root_nodes_.empty()) freeze();
    std::vector<T> result;
    for (const auto& item : root_nodes_) sort(item, result);
    return result;
  }

  void sort(const T& name, std::vector<T>& result) {
    bool can_sort = true;
    for (const auto& item : previous_nodes_[name]) {
      if (std::find(result.begin(), result.end(), item) == result.end()) {
        can_sort = false;
      }
    }
    if (!can_sort) {
      return;
    }
    result.push_back(name);
    if (!next_nodes_[name].empty()) {
      for (const auto& item : next_nodes_[name]) {
        if (std::find(result.begin(), result.end(), item) == result.end()) sort(item, result);
      }
    }
  }

  void freeze() {
    for (const auto& item : next_nodes_) {
      for (const auto& item_next : item.second) {
        if (next_nodes_.find(item_next) == next_nodes_.end()) {
          throw std::invalid_argument("next: node " + item_next + " not exists");
        }
      }
      all_nodes_.insert(item.first);
      all_nodes_.insert(item.second.begin(), item.second.begin());
    }
    for (const auto& item : next_nodes_) {
      all_nodes_.insert(item.first);
      all_nodes_.insert(item.second.begin(), item.second.begin());
    }
    root_nodes_ = all_nodes_;
    for (const auto& item : next_nodes_) {
      for (const auto& item_next : item.second) {
        root_nodes_.erase(item_next);

        auto iter_previous = previous_nodes_.find(item_next);
        if (iter_previous == previous_nodes_.end()) {
          previous_nodes_[item_next] = std::set<std::string>();
          previous_nodes_[item_next].insert(item.first);
        } else {
          previous_nodes_[item_next].insert(item.first);
        }
      }
    }

    for (const auto& item : root_nodes_) {
      previous_nodes_[item] = std::set<std::string>();
    }
  }

  void sub_graph_start(const T& start_node_name) { subgraph_start_.push(start_node_name); }
  std::set<T> sub_graph_end(const T& end_node_name, T& start_name_out) {
    if (all_nodes_.empty()) freeze();
    if (subgraph_start_.size() > 1) {
      subgraph_start_.pop();
      return std::set<T>();
    } else if (subgraph_start_.empty()) {
      return std::set<T>();
      throw std::invalid_argument("PipelineV3: start node and end node not match. end =  " +
                                  end_node_name);
    }
    const T start_node_name = subgraph_start_.top();
    subgraph_start_.pop();
    start_name_out = start_node_name;
    if (start_node_name == end_node_name) {
      return std::set<T>();
    }

    std::set<T> results{start_node_name};

    auto nexts = next_nodes_.at(start_node_name);

    while (true) {
      if (nexts.empty()) break;
      results.insert(nexts.begin(), nexts.end());
      //   if (results.count(end_node_name) != 0)
      //     break;
      std::set<T> nexts_nexts;
      for (const auto& data : nexts) {
        if (data == end_node_name) continue;
        nexts_nexts.insert(next_nodes_[data].begin(), next_nodes_[data].end());
      }
      std::swap(nexts_nexts, nexts);
    }

    std::set<T> previous_results{end_node_name};
    auto previous = previous_nodes_.at(end_node_name);
    while (true) {
      if (previous.empty()) break;
      previous_results.insert(previous.begin(), previous.end());
      //   if (results.count(end_node_name) != 0)
      //     break;
      std::set<T> previous_previous;
      for (const auto& data : previous) {
        if (data == start_node_name) continue;
        previous_previous.insert(previous_nodes_[data].begin(), previous_nodes_[data].end());
      }
      std::swap(previous_previous, previous);
    }
    if (true || previous_results == results) {
      subgraphs_[start_name_out] = end_node_name;
      sub_nodes_[start_name_out] = results;
      return results;
    } else {
      results.clear();
      return results;
    }
  }
  //   const std::set<T>& next(const T& node_name) {}
  //   const std::set<T>& previous(const T& node_name) {}

  //   const Graph& sub_graph(const T& start, const T& end) {}
  //   const Graph& sub_graph(const T& start) {}
  //   bool start_of_sub_graph(const T& node_name) {}
  //   const T& end(const T& node_name) {}

  //   bool is_subgraph(const T& start, const T& end);
  std::unordered_map<T, T>& subgraphs() { return subgraphs_; }

  bool is_PipelineV3_start(const T& node_name) {
    return subgraphs_.find(node_name) != subgraphs_.end();
  }

  T get_PipelineV3_end(const T& node_name) {
    auto iter = subgraphs_.find(node_name);
    return iter->second;
  }
  const std::set<T>& get_PipelineV3_nodes(const T& node_name) {
    auto iter = sub_nodes_.find(node_name);
    return iter->second;
  }

  const std::set<T>& next(const T& node_name) { return next_nodes_[node_name]; }

  std::set<T> nexts(const T& node_name) {
    std::set<T> results{};

    auto nexts = next_nodes_.at(node_name);

    while (true) {
      if (nexts.empty()) break;
      results.insert(nexts.begin(), nexts.end());
      //   if (results.count(end_node_name) != 0)
      //     break;
      std::set<T> nexts_nexts;
      for (const auto& data : nexts) {
        nexts_nexts.insert(next_nodes_[data].begin(), next_nodes_[data].end());
      }
      std::swap(nexts_nexts, nexts);
    }
    return results;
  }

  std::set<T> nexts(const T& node_name, const T& end_name) {
    std::set<T> results{};

    auto nexts = next_nodes_.at(node_name);

    while (true) {
      if (nexts.empty()) break;

      results.insert(nexts.begin(), nexts.end());
      //   if (results.count(end_node_name) != 0)
      //     break;
      std::set<T> nexts_nexts;
      for (const auto& data : nexts) {
        if (data == end_name) continue;
        nexts_nexts.insert(next_nodes_[data].begin(), next_nodes_[data].end());
      }
      std::swap(nexts_nexts, nexts);
    }
    return results;
  }

  bool is_root(const T& node_name) {
    if (root_nodes_.empty()) {
      freeze();
    }
    return root_nodes_.count(node_name) != 0;
  }

  bool is_leaf(const T& node_name) {
    if (root_nodes_.empty()) {
      freeze();
    }
    return next_nodes_.at(node_name).empty();
  }

 private:
  std::stack<T> subgraph_start_;
  std::unordered_map<T, std::unordered_map<std::string, T>> config_;
  std::unordered_map<T, T> subgraphs_;
  std::set<T> root_nodes_;
  std::set<T> all_nodes_;
  std::unordered_map<T, std::set<T>> previous_nodes_;
  std::unordered_map<T, std::set<T>> next_nodes_;
  std::unordered_map<T, std::set<T>> sub_nodes_;
};

}  // namespace ipipe
