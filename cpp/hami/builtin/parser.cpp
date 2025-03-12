#include <queue>
#include <stdexcept>

#include "hami/builtin/parser.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/helper/string.hpp"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"

namespace hami::parser {

void broadcast_global(str::mapmap& config) {
    auto iter = config.find(TASK_GLOBAL_KEY);
    if (iter == config.end()) return;

    // only global
    if (config.size() == 1) {
        config[TASK_DEFAULT_NAME_KEY] = iter->second;
        return;
    }
    const str::str_map& global = iter->second;
    for (auto& item : config) {
        if (item.first != TASK_GLOBAL_KEY) {
            for (const auto& global_item : global) {
                if (item.second.find(global_item.first) == item.second.end()) {
                    item.second[global_item.first] = global_item.second;
                }
            }
        }
    }
}

std::unordered_set<std::string> set_node_name(str::mapmap& config) {
    std::unordered_set<std::string> node_names;
    for (auto& item : config) {
        if (item.first != TASK_GLOBAL_KEY) {
            item.second[TASK_NODE_NAME_KEY] = item.first;
            node_names.insert(item.first);
        }
    }

    return node_names;
}

str::str_map get_global_config(const str::mapmap& config) {
    if (config.find(TASK_GLOBAL_KEY) == config.end()) {
        if (config.size() == 1) {
            return config.begin()->second;
        }
        return str::str_map();
    } else {
        return config.at(TASK_GLOBAL_KEY);
    }
}

size_t count(const str::mapmap& config) {
    return config.find(TASK_GLOBAL_KEY) == config.end() ? config.size()
                                                        : config.size() - 1;
}

DagParser::DagParser(const str::mapmap& config) {
    for (const auto& item : config) {
        if (item.first == TASK_GLOBAL_KEY) {
            continue;
        }
        Node node_config;

        // handle next
        auto iter = item.second.find(TASK_NEXT_KEY);
        if (iter != item.second.end()) {
            auto nexts = str::str_split(iter->second, ',');
            node_config.next.insert(nexts.begin(), nexts.end());
        }
        node_config.cited = node_config.next;

        // handle or
        iter = item.second.find(TASK_OR_KEY);
        if (iter != item.second.end()) {
            node_config.or_filter = iter->second != "0";
        }

        // handle map
        iter = item.second.find(TASK_MAP_KEY);
        if (iter != item.second.end()) {
            node_config.map_config = parse_map_config(iter->second);
        }

        dag_config_[item.first] = node_config;
    }

    // update previous
    update_previous();

    // get all roots node through previous
    update_roots();

    update_cited_from_map();

    // // if a node has been cited by multiple nodes(with next/map), it's data
    // must be acquired by deep copy
    for (const auto& item : dag_config_) {
        if (item.second.cited.size() > 1) {
            for (const auto& cited_item : item.second.cited) {
                if (dag_config_[cited_item].map_config.empty()) {
                    SPDLOG_INFO(
                        "DagParser: node `" + item.first +
                        "` has been cited(through map or next) by more than "
                        "one node, but one "
                        "of them - " +
                        cited_item +
                        " has no map config, default set to [result:data]");
                    dag_config_[cited_item].map_config = str::mapmap();
                    dag_config_[cited_item].map_config[item.first] =
                        str::str_map{{TASK_RESULT_KEY, TASK_DATA_KEY}};
                }
            }
        }
    }

    // check or
    for (const auto& item : dag_config_) {
        if (item.second.cited.size() > 1) {
            HAMI_ASSERT(!item.second.or_filter,
                        "DagParser: `map` and `or` cannot be used together");
        }
    }

    // try sort
    try_topological_sort();
}

void DagParser::update_previous() {
    for (auto& item : dag_config_) {
        for (const auto& next : item.second.next) {
            auto iter = dag_config_.find(next);
            HAMI_ASSERT(iter != dag_config_.end(),
                        "DagParser: next node not found: " + next);
            // item.first => next
            iter->second.previous.insert(item.first);
        }
        if (item.second.map_config.empty()) {
            HAMI_ASSERT(item.second.previous.size() <= 1,
                        "DagParser: " + item.first +
                            " has more than "
                            "one previous node but no map config");
        }
    }
}

void DagParser::update_roots() {
    roots_.clear();
    for (auto& item : dag_config_) {
        if (item.second.previous.empty()) {
            roots_.insert(item.first);
        }
    }
    HAMI_ASSERT(!roots_.empty());
}

void DagParser::update_cited_from_map() {
    for (auto& map_dst : dag_config_) {
        for (const auto& dual_map : map_dst.second.map_config) {
            dag_config_[dual_map.first].cited.insert(map_dst.first);
        }
    }
}

std::vector<std::string> DagParser::try_topological_sort() {
    std::vector<std::string> result;
    std::unordered_map<std::string, int> in_degree;
    std::queue<std::string> q;

    // 初始化入度
    for (const auto& item : dag_config_) {
        in_degree[item.first] = item.second.previous.size();
        if (in_degree[item.first] == 0) {
            q.push(item.first);
        }
    }

    // 拓扑排序
    while (!q.empty()) {
        std::string current = q.front();
        q.pop();
        result.push_back(current);

        for (const auto& next : dag_config_[current].next) {
            in_degree[next]--;
            if (in_degree[next] == 0) {
                q.push(next);
            }
        }
    }

    // 检查是否存在环
    if (result.size() != dag_config_.size()) {
        throw std::runtime_error("Graph contains a cycle");
    }

    return result;
}

str::mapmap DagParser::parse_map_config(const std::string& config) {
    str::mapmap map_config;
    auto maps = str::items_split(config, ',');
    for (const auto& maps_item : maps) {
        str::str_map in_map;
        auto maps_data = str::flatten_brackets(maps_item);
        HAMI_ASSERT(maps_data.size() == 2,
                    "map should be in the form of node_name[src_key:dst_key]");
        in_map = str::map_split(maps_data[1], ':', ',', "");
        if (!in_map.empty()) {
            map_config[maps_data[0]] = in_map;
        }
    }
    return map_config;
}

std::unordered_set<std::string> DagParser::get_subgraph(
    const std::string& root) {
    // std::unordered_set<std::string> re;
    HAMI_ASSERT(dag_config_.find(root) != dag_config_.end(),
                "DagParser: " + root + " not found");
    HAMI_ASSERT(dag_config_[root].previous.empty(),
                root + " is not a root node");

    std::unordered_set<std::string> parsered;

    std::unordered_set<std::string> not_parsered{root};

    while (!not_parsered.empty()) {
        auto iter_next = *not_parsered.begin();
        not_parsered.erase(iter_next);
        parsered.insert(iter_next);
        for (const auto& item : dag_config_[iter_next].next) {
            if (parsered.find(item) == parsered.end()) {
                not_parsered.insert(item);
            } else {
                not_parsered.erase(item);
            }
        }
    }

    // check independency
    for (const auto& item : parsered) {
        for (const auto& pre : dag_config_[item].previous) {
            HAMI_ASSERT(parsered.find(pre) != parsered.end(),
                        "DagParser: " + pre + " is not in subgraph");
        }
    }
    return parsered;
}

dict DagParser::prepare_data_from_previous(
    const std::string& node, std::unordered_map<std::string, dict>& processed) {
    if (dag_config_.at(node).map_config.empty()) {
        auto previous_node = *dag_config_.at(node).previous.begin();
        auto re = processed.at(previous_node);
        if (re->find(TASK_RESULT_KEY) == re->end()) {
            if (!dag_config_.at(node).or_filter) {
                throw std::runtime_error("DagParser: " + previous_node +
                                         " has no result");
            }
        } else {
            if (dag_config_.at(node).or_filter) {
                processed[node] = re;
                return re;
            } else {
                (*re)[TASK_DATA_KEY] = (*re)[TASK_RESULT_KEY];
            }
        }
        return re;
    }

    dict re = make_dict();
    dict src_dict_context;
    bool has_context_already{false};
    for (const auto& item : dag_config_.at(node).map_config) {
        const auto& src_node = item.first;
        const auto& src_dict = processed.at(src_node);
        for (const auto& [src_key, dst_key] : item.second) {
            auto iter = src_dict->find(src_key);
            HAMI_ASSERT(iter != src_dict->end(),
                        "DagParser: " + src_key + " not found in " + src_node);
            (*re)[dst_key] = iter->second;
            if (src_key == TASK_CONTEXT_KEY || dst_key == TASK_CONTEXT_KEY) {
                has_context_already = true;
            }
        }
        if (src_dict->find(TASK_CONTEXT_KEY) != src_dict->end()) {
            src_dict_context = src_dict;
        }
    }
    if (!has_context_already && src_dict_context) {
        (*re)[TASK_CONTEXT_KEY] = src_dict_context->at(TASK_CONTEXT_KEY);
    }

    HAMI_ASSERT(
        re->find(TASK_DATA_KEY) != re->end(),
        std::string("DagParser: ") + TASK_DATA_KEY + " not found in target");
    return re;
}
}  // namespace hami::parser
