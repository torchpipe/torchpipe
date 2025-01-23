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
#include <set>
#include <queue>
#include <unordered_map>
#include "dict.hpp"
#include "ipipe_common.hpp"
#include <algorithm>
namespace ipipe {

namespace {
void borrow_context(dict src, dict dst) {
  auto iter = src->find(TASK_CONTEXT_KEY);
  if (iter == src->end()) {
    return;
  } else {
    auto iter_dst = dst->find(TASK_CONTEXT_KEY);
    if (iter_dst == dst->end()) {
      (*dst)[TASK_CONTEXT_KEY] = iter->second;
    }
  }
}
}  // namespace
// todo: SFINAE
class LogicalGraph {
 public:
  using MapSrc = std::string;
  const std::unordered_map<std::string, std::string>& get_fold_map() { return fold_map_; };

 private:
  struct LogicalNodeConfig {
    int indegree{0};
    std::set<std::string> next;
    std::set<std::string> previous;

    std::unordered_map<std::string, std::unordered_map<std::string, MapSrc>> map_reduces;
    std::unordered_map<std::string, MapSrc> map_reduce;
    std::set<std::string> references;
  };

 private:
  std::unordered_map<std::string, LogicalNodeConfig>
      node_configs_;  // 记录每个顶点的入度
                      //   std::unordered_map<T, int> key_map_;
  std::vector<std::string> sortted_;
  std::set<std::string> roots_;
  std::unordered_map<std::string, std::set<std::string>> sub_tree_;

  std::unordered_map<std::string, std::string> fold_map_;
  bool split();

  void set_fold_map(const std::unordered_map<std::string, std::string>& fold_map) {
    fold_map_ = fold_map;
  }

 public:
  bool check_map();
  bool update_map(
      const std::string& node_name,
      const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>&
          map_reduces,
      const std::unordered_map<std::string, std::string>& map_reduce) {
    auto iter = node_configs_.find(node_name);
    if (iter == node_configs_.end()) return false;
    auto& config = iter->second;
    config.map_reduce = map_reduce;
    config.map_reduces = map_reduces;
    finalize();
    check_map();
    return true;
  }

  bool get_serial_nexts(std::string node_name, std::set<std::string>& nexts) {
    // std::set<std::string> sub_graph;
    auto iter = node_configs_.find(node_name);
    if (iter == node_configs_.end()) return false;
    while (true) {
      if (iter->second.next.size() == 0)
        break;
      else if (iter->second.next.size() == 1) {
        node_name = *iter->second.next.begin();
        // sub_graph.insert(node_name);
        iter = node_configs_.find(node_name);
        assert(iter != node_configs_.end());
        if (1 != iter->second.previous.size()) {
          nexts.insert(node_name);
          break;
        }
      } else {
        nexts.insert(iter->second.next.begin(), iter->second.next.end());
        break;
      }
    };
    return true;
  }

  std::set<std::string> get_all() {
    std::set<std::string> sub_graph;
    for (const auto& item : node_configs_) {
      sub_graph.insert(item.first);
    }
    return sub_graph;
  }

  bool get_subgraph_nexts(std::string node_name, std::set<std::string>& nexts) {
    auto iter = node_configs_.find(node_name);
    if (iter == node_configs_.end()) return false;

    std::set<std::string> waiting_nodes = get_subtree(node_name);

    std::set<std::string> should_remove{node_name};
    while (!should_remove.empty()) {
      for (const auto& item : should_remove) waiting_nodes.erase(item);
      should_remove.clear();

      for (const auto& node : waiting_nodes) {
        for (const auto& item : node_configs_[node].previous) {
          if (waiting_nodes.count(item) == 0 && item != node_name) {
            should_remove.insert(node);
            break;  // next node
          }
        }
      };
    }
    waiting_nodes.insert(node_name);
    for (const auto& node : waiting_nodes) {
      for (const auto& item : node_configs_[node].next) {
        if (waiting_nodes.count(item) == 0) {
          nexts.insert(item);
        }
      }
    }
    return true;
  }

  void del_relation(const std::set<std::string>& del_nodes) {
    for (const auto& del_item : del_nodes) {
      auto iter = node_configs_.find(del_item);
      IPIPE_ASSERT(iter != node_configs_.end());
      auto& config = iter->second;
      config.map_reduce.clear();
      config.map_reduces.clear();
    }
  }

  void del_nodes(const std::set<std::string>& del_nodes) {
    for (const auto& del_item : del_nodes) {
      auto iter = node_configs_.find(del_item);
      IPIPE_ASSERT(iter != node_configs_.end());
      node_configs_.erase(iter);
    }
  }

  std::shared_ptr<LogicalGraph> clone() { return std::make_shared<LogicalGraph>(*this); }
  std::shared_ptr<LogicalGraph> update(
      const std::unordered_map<std::string, std::set<std::string>>& next_config,
      const std::set<std::string>& del_nodes,
      std::unordered_map<std::string, std::string>& fold_map) {
    auto original_root = get_roots();
    auto new_graph = clone();
    new_graph->del_relation(del_nodes);
    const auto& old_roots = extract_dependent_roots_from_next();
    if (!new_graph->update_impl(next_config, del_nodes, old_roots)) {
      assert(false);
      return nullptr;
    };

    if (!new_graph->finalize()) {
      assert(false);
      return nullptr;
    }
    const auto& new_root = new_graph->get_roots();

    std::vector<std::string> fold;
    std::set_difference(new_root.begin(), new_root.end(), original_root.begin(),
                        original_root.end(), std::back_inserter<std::vector<std::string>>(fold));

    // 添加 【新增独立节点->输入点】 映射
    for (const auto& fold_node : fold) {
      if (del_nodes.count(fold_node) == 1) {
        continue;
      }
      for (const auto& item : next_config) {
        if (this->sub_tree_[item.first].count(fold_node) == 1) {
          auto iter = fold_map.find(fold_node);
          if (iter == fold_map.end()) {
            fold_map[fold_node] = item.first;
          } else {
            // 新增独立节点在原始图中只能连通到一个输入点
            assert(false);
            return nullptr;
          }
        }
      }
      // 新增独立节点在原始图中必须是跳跃点的子节点
      auto iter = fold_map.find(fold_node);
      if (iter == fold_map.end()) {
        assert(false);
        return nullptr;
      }
    }

    // 跳跃点不能是新增独立节点
    for (const auto& item : next_config) {
      if (fold_map.find(item.first) != fold_map.end()) {
        assert(false);
        return nullptr;
      }
    }
    new_graph->set_fold_map(fold_map);
    if (!new_graph->check_map()) {
      assert(false);
      return nullptr;
    }
    return new_graph;
  }

  std::shared_ptr<LogicalGraph> as_root(std::string node_name) {
    auto new_graph = clone();
    auto previous = new_graph->get_all_previous(node_name);
    // new_graph->del_relation(previous);
    new_graph->del_relation({node_name});
    new_graph->del_nodes(previous);

    if (!new_graph->finalize()) {
      return nullptr;
    }

    return new_graph;
  }

  bool update_impl(const std::unordered_map<std::string, std::set<std::string>>& next_config,
                   const std::set<std::string>& del_nodes, const std::set<std::string>& old_roots) {
    for (const auto& item : next_config) {
      auto iter = node_configs_.find(item.first);
      if (iter == node_configs_.end()) {
        assert(false);
        return false;
      }
      iter->second.next = item.second;
    }

    // delete relation
    for (const auto& item : del_nodes) {
      auto iter = node_configs_.find(item);
      if (iter == node_configs_.end()) {
        assert(false);
        return false;
      }
      iter->second.next = std::set<std::string>();
    }

    for (const auto& item : del_nodes) {
      auto iter = node_configs_.find(item);
      if (iter == node_configs_.end()) {
        assert(false);
        return false;
      }
      if (iter->second.next.count(item) == 1) {
        iter->second.next.erase(item);
      };
    }

    const auto& new_roots = extract_dependent_roots_from_next();
    std::set<std::string> new_added_roots;
    for (const auto& item : new_roots) {
      if (old_roots.count(item) == 0) {
        new_added_roots.insert(item);
      }
    }
    std::unordered_map<std::string, std::set<std::string>> new_config;
    for (const auto& item : new_added_roots) {
      new_config[item] = std::set<std::string>();
    }
    if (!new_config.empty()) return update_impl(new_config, std::set<std::string>(), new_roots);
    return true;
  }

  LogicalGraph() = default;

  LogicalGraph& add(const std::string& v, const std::string& w) {
    if (node_configs_.find(v) == node_configs_.end()) {
      node_configs_[v] = LogicalNodeConfig();
    }
    if (node_configs_.find(w) == node_configs_.end()) {
      node_configs_[w] = LogicalNodeConfig();
    }

    node_configs_[v].next.insert(w);
    // node_configs_[w].previous.insert(v);
    // ++node_configs_[w].indegree;
    return *this;
  }

  LogicalGraph& add(const std::string& v) {
    if (node_configs_.find(v) == node_configs_.end()) {
      node_configs_[v] = LogicalNodeConfig();
    };
    return *this;
  }

  LogicalGraph& add(const std::string& v, const std::vector<std::string>& ws) {
    for (const auto& w : ws) {
      add(v, w);
    }

    return *this;
  }

  LogicalGraph& set_map_reduce(const std::string& v, const std::string& config, bool is_reduce);
  bool check();

  bool finalize();
  std::set<std::string> extract_dependent_roots_from_next();
  std::set<std::string> get_all_previous(std::string node);
  std::string get_default_filter(const std::string& node_name) const;
  const std::set<std::string>& get_subtree(const std::string& node_name) const {
    auto iter = sub_tree_.find(node_name);
    assert(iter != sub_tree_.end());
    return iter->second;
  }

  const std::vector<std::string>& get_sortted() const { return sortted_; }
  const std::set<std::string>& get_roots() const { return roots_; }
  bool is_root(const std::string& node_name) const {
    return roots_.find(node_name) != roots_.end();
  }

  bool is_valid(const std::string& node_name) const {
    return node_configs_.find(node_name) != node_configs_.end();
  }
  const std::set<std::string>& get_previous(const std::string& node_name) const noexcept {
    auto iter = node_configs_.find(node_name);
    assert(iter != node_configs_.end());
    return iter->second.previous;
  }

  const std::set<std::string>& get_next(const std::string& node_name) const noexcept {
    auto iter = node_configs_.find(node_name);
    assert(iter != node_configs_.end());
    return iter->second.next;
  }

  void map_data(const std::string& node_name, dict src, dict target,
                const std::unordered_map<std::string, MapSrc>& config) const {
    if (!src || !target) throw std::invalid_argument("invalid src or target.");
    for (const auto& item : config) {
      auto iter_src = src->find(item.second);
      if (iter_src == src->end()) {
        throw std::out_of_range(node_name + ": can't find key `" + item.second + "` from source.");
      }
      (*target)[item.first] = iter_src->second;
    }
  }

  dict get_mapped_previous(const std::string& node_name,
                           const std::unordered_map<std::string, dict>& processed) const {
    auto iter = node_configs_.find(node_name);
    assert(iter != node_configs_.end());
    const auto& pre = iter->second.previous;
    IPIPE_ASSERT(!pre.empty(), "map is not supported for root node.");
    dict curr_data;
    dict previous_data;
    if (!iter->second.map_reduce.empty()) {
      // IPIPE_ASSERT(pre.size() == 1);
      std::string pre_key;
      for (const auto& item : pre) {
        auto iter_pre = processed.find(item);
        if (iter_pre != processed.end()) {
          previous_data = iter_pre->second;
          break;
        }
      }
      if (!previous_data) {
        throw std::runtime_error(node_name + ": invalid previous node");
      }
      curr_data = make_dict(node_name);
      map_data(node_name, previous_data, curr_data, iter->second.map_reduce);
      borrow_context(previous_data, curr_data);

    } else if (!iter->second.map_reduces.empty()) {
      curr_data = make_dict(node_name);
      for (const auto& node_config : iter->second.map_reduces) {
        auto iter_pre = processed.find(node_config.first);
        if (iter_pre == processed.end()) {
          throw std::runtime_error("invalid previous node: " + node_config.first);
        }
        map_data(node_name, iter_pre->second, curr_data, node_config.second);
        borrow_context(iter_pre->second, curr_data);
      }
    } else {
      if (pre.size() != 1) {
        IPIPE_ASSERT(pre.size() == 1, node_name + ": unmatch map.");
      }
      const auto& pre_node = *pre.begin();
      auto iter_pre = processed.find(pre_node);
      IPIPE_ASSERT(iter_pre != processed.end());
      previous_data = iter_pre->second;
      IPIPE_ASSERT(previous_data);

      auto iter = node_configs_.find(pre_node);
      IPIPE_ASSERT(iter != node_configs_.end());

      if (iter->second.references.count(pre_node) == 1) {
        (*previous_data)["node_name"] = node_name;
        return previous_data;
      } else {
        curr_data = make_dict(node_name, previous_data);
      }
      // for (const auto& item : pre) {
      //   auto iter_pre = processed.find(item);
      //   if (iter_pre != processed.end()) {
      //     previous_data = iter_pre->second;
      //     const auto& ne = get_next(item);
      //     if (false && ne.size() == 1) {  // todo
      //       (*previous_data)["node_name"] = node_name;
      //       return previous_data;
      //     } else if (true || ne.size() > 1) {  // todo
      //       curr_data = make_dict(node_name, previous_data);
      //       // borrow_context(previous_data, curr_data);
      //     } else {
      //       throw std::runtime_error("invalid next size=" + std::to_string(ne.size()));
      //     }
      //     break;
      //   }
      // }
    }
    if (curr_data->find(TASK_DATA_KEY) == curr_data->end()) {
      throw std::runtime_error("map: no TASK_DATA_KEY contained in the target dict");
    }
    return curr_data;
  }
};
};  // namespace ipipe