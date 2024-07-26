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

#include "config_parser.hpp"

#include "base_logging.hpp"
#include "params.hpp"
#include "toml.hpp"
#include "dict.hpp"
#include "ipipe_common.hpp"
namespace ipipe {

std::string toml2str(toml::value v) {
  switch (v.type()) {
    case toml::value_t::boolean: {
      return std::to_string(v.as_boolean());
    }
    case toml::value_t::integer: {
      return std::to_string(v.as_integer());
    }
    case toml::value_t::floating: {
      return std::to_string(v.as_floating());
    }
    case toml::value_t::string: {
      return std::string(v.as_string());
    }
    case toml::value_t::offset_datetime: {
      return std::to_string(v.as_offset_datetime());
    }
    case toml::value_t::local_datetime: {
      return std::to_string(v.as_local_datetime());
    }
    case toml::value_t::local_date: {
      return std::to_string(v.as_local_date());
    }
    case toml::value_t::local_time: {
      // return std::to_string(v.as_local_time());
    }
    case toml::value_t::array: {
      // return std::to_string(v.as_array());
    }
    case toml::value_t::table:
      // return std::to_string(v.as_table());

    case toml::value_t::empty:

    default:
      throw std::runtime_error("unsupported toml type.");
  }
};

void update_global(mapmap& config) {
  // 将 config["global"] 合并到其他键值中
  auto iter_global = config.find("global");
  if (iter_global == config.end()) return;
  for (auto iter = config.begin(); iter != config.end(); ++iter) {
    if (iter == iter_global) {
      continue;
    }
    auto new_config = iter_global->second;

    for (auto iter_config = iter->second.begin(); iter_config != iter->second.end();
         ++iter_config) {
      new_config[iter_config->first] = iter_config->second;
    }
    std::swap(iter->second, new_config);
  }
}

void update_node(std::unordered_map<std::string, std::string>& single_node, toml::key key,
                 const toml::value& value, mapmap& results) {
  auto config = value.as_table();

  for (auto iter = config.begin(); iter != config.end(); ++iter) {
    if (iter->second.is_table()) {
      std::unordered_map<std::string, std::string> single_node_inner;
      single_node_inner["node_name"] = key + "." + iter->first;

      // auto config_inner = iter->second.as_table();
      update_node(single_node_inner, iter->first, iter->second, results);
      results[key + "." + iter->first] = single_node_inner;
    } else {
      IPIPE_CHECK(!is_reserved(iter->first), "`" + iter->first + "` is reserved word.");
      single_node[iter->first] = toml2str(iter->second);
    }
  }
}
/// 从toml文件中解析配置
mapmap parse_toml_data(const std::unordered_map<toml::key, toml::value>& data) {
  mapmap results;
  if (data.empty()) return results;

  // 将toml保存到 results； 对于没有节点名称的项， 放入global键值中
  results["global"] = std::unordered_map<std::string, std::string>();
  for (const auto& item : data) {
    if (item.second.is_table()) {
      std::unordered_map<std::string, std::string> single_node;
      single_node["node_name"] = item.first;

      update_node(single_node, item.first, item.second, results);

      results[item.first] = single_node;
    } else {
      results["global"][item.first] = toml2str(item.second);
      if (item.first == "node_name")
        throw std::runtime_error("Error: use private key: " + item.first + " is not allowed");
    }
  }

  return results;
};

mapmap parse_toml(std::string toml_path) {
  auto data = toml::parse(toml_path);
  return parse_toml_data(data.as_table());
}

std::set<std::string> handle_ring(const mapmap& config_param) { return std::set<std::string>(); }
/**
 * @brief 解析全局配置，设置默认参数，中括号语法解析，对于单节点配置默认节点名称
 *
 * @param config
 * @todo 把默认参数的设置拆分到其他地方
 */
void handle_config(mapmap& config) {
  for (auto& item : config) {
    for (auto& inner : item.second) {
      replace_task_key(inner.second);
    }
  }

  if (config.size() == 1) {
    auto iter = config.find("global");
    if (iter != config.end()) {
      brackets_split(iter->second, "Interpreter::backend");
      config["node_name_none"] = iter->second;
    } else {
      brackets_split(config.begin()->second, "Interpreter::backend");
    }
  }
  for (auto& item : config) {
    if (item.first == "global") {
      auto iter = item.second.find("Interpreter::backend");
      if (iter != item.second.end()) {
        brackets_split(item.second["Interpreter::backend"], item.second);
      }
    }

    item.second["node_name"] = item.first;
    auto iter = item.second.find("backend");
    if (iter == item.second.end()) {
      // default to Identity
      item.second["backend"] = "Identity";
    } else {
      // 处理中括号， 将backend=A[B[C]]展开为 backend=A， A::backend=B,
      // B::backend=C
      brackets_split(item.second["backend"], item.second);
    }
  }
}
}  // namespace ipipe