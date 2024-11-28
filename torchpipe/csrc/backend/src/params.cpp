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

#include "params.hpp"
#include <algorithm>
#include <regex>
#include "base_logging.hpp"
#include "dict.hpp"
#include "ipipe_common.hpp"
#include "exception.hpp"
#include "ipipe_utils.hpp"

namespace ipipe {

void replace_task_key(std::string& key) {
  // if (startswith(key, "TASK_") && endswith(key, "_KEY")) {
  //   const auto iter = TASK_KEY_MAP.find(key);
  //   if (iter != TASK_KEY_MAP.end()) key = iter->second;
  // }

  do {
    auto iter_start = key.find("TASK_");
    auto iter_end = key.find("_KEY");

    if (iter_start != std::string::npos && iter_end != std::string::npos && iter_start < iter_end) {
      iter_end += sizeof("_KEY") - 1;
      std::string result = key.substr(iter_start, iter_end - iter_start);
      const auto iter = TASK_KEY_MAP.find(result);
      if (iter != TASK_KEY_MAP.end()) {
        key = key.substr(0, iter_start) + iter->second + key.substr(iter_end);
      } else {
        throw std::invalid_argument("cann't parse " + result);
      }
    } else {
      break;
    }
  } while (true);
}

void replace_all(std::string& s, std::string const& toReplace, std::string const& replaceWith) {
  std::ostringstream oss;
  std::size_t pos = 0;
  std::size_t prevPos = pos;

  while (true) {
    prevPos = pos;
    pos = s.find(toReplace, pos);
    if (pos == std::string::npos) break;
    oss << s.substr(prevPos, pos - prevPos);
    oss << replaceWith;
    pos += toReplace.size();
  }

  oss << s.substr(prevPos);
  s = oss.str();
}
std::string str_join(std::vector<std::string> datas, char sp) {
  if (datas.size() < 1)
    return "";
  else if (datas.size() == 1)
    return datas[0];
  else {
    std::string result = datas[0];
    for (std::size_t i = 1; i < datas.size(); ++i) {
      result += sp;
      result += datas[i];
    }
    return result;
  }
}

std::string str_join(std::vector<std::string> datas) {
  if (datas.size() < 1)
    return "";
  else if (datas.size() == 1)
    return datas[0];
  else {
    std::string result = datas[0];
    for (std::size_t i = 1; i < datas.size(); ++i) {
      result += datas[i];
    }
    return result;
  }
}

void replace_all(std::string& s, std::string& original, std::string const& toReplace,
                 std::string const& replaceWith, const std::set<std::string>& valid_node,
                 std::string reduce_node_name, const std::set<std::string>& reduce_nodes) {
  assert(s == original);
  // auto data = str_split(s, ',');
  auto data = str_split_brackets_match(s, ',', '[', ']');
  std::vector<std::string> result;
  std::vector<std::string> original_result;
  for (auto& item : data) {
    auto iter = item.find("[");
    assert(iter != std::string::npos);

    auto name = item.substr(0, iter);
    if (valid_node.count(name) != 0) {
      replace_all(item, toReplace, replaceWith);
      result.push_back(item);
    } else {
      original_result.push_back(item);
    }
  }
  s = str_join(result, ',');
  reduce_node_name += "[";
  uint32_t index = 0;
  for (const auto& item : reduce_nodes) {
    if (item == TASK_DATA_KEY) {
      reduce_node_name += TASK_RESULT_KEY;
      reduce_node_name += ":";
      reduce_node_name += item;
    } else {
      reduce_node_name += item;
      reduce_node_name += ":";
      reduce_node_name += item;
    }
    if (index++ != reduce_nodes.size() - 1)
      reduce_node_name += ",";
    else {
      reduce_node_name += "]";
    }
  }
  original_result.emplace_back(reduce_node_name);
  original = str_join(original_result, ',');
}

std::vector<std::string> str_split_brackets_match(std::string strtem, char a, char left,
                                                  char right) {
  std::vector<std::string> strvec;
  if (strtem.empty()) return strvec;

  auto itor = std::remove(strtem.begin(), strtem.end(), ' ');
  strtem.erase(itor, strtem.end());

  std::string::size_type pos1, pos2;
  pos2 = strtem.find(a);
  pos1 = 0;
  while (std::string::npos != pos2) {
    int brackets_num = 0;
    for (auto i = pos1; i < pos2; ++i) {
      if (strtem[i] == left)
        brackets_num++;
      else if (strtem[i] == right) {
        if (0 > --brackets_num) {
          SPDLOG_ERROR("unmatch brackets: " + strtem);
          throw std::invalid_argument("unmatch brackets: " + strtem);
        }
      }
    }
    if (brackets_num == 0) {
      strvec.push_back(strtem.substr(pos1, pos2 - pos1));

      pos1 = pos2 + 1;
      pos2 = strtem.find(a, pos1);
    } else {
      pos2 = strtem.find(a, pos2 + 1);
    }
  }
  strvec.push_back(strtem.substr(pos1));
  for (auto iter_vec = strvec.begin(); iter_vec != strvec.end();) {
    if (iter_vec->empty())
      iter_vec = strvec.erase(iter_vec);
    else
      ++iter_vec;
  }
  return strvec;
}
std::vector<std::string> str_split(std::string strtem, char a, bool keep_empty_result) {
  std::vector<std::string> strvec;
  if (strtem.empty() && keep_empty_result) return {""};

  auto itor = std::remove(strtem.begin(), strtem.end(), ' ');
  strtem.erase(itor, strtem.end());

  std::string::size_type pos1, pos2;
  pos2 = strtem.find(a);
  pos1 = 0;
  while (std::string::npos != pos2) {
    strvec.push_back(strtem.substr(pos1, pos2 - pos1));

    pos1 = pos2 + 1;
    pos2 = strtem.find(a, pos1);
  }
  strvec.push_back(strtem.substr(pos1));
  for (auto iter_vec = strvec.begin(); iter_vec != strvec.end();) {
    if (iter_vec->empty()) {
      if (!keep_empty_result)
        iter_vec = strvec.erase(iter_vec);
      else {
        SPDLOG_WARN("empty split: {}", strtem);
        ++iter_vec;
      }

    } else
      ++iter_vec;
  }
  return strvec;
}
std::vector<float> strs2number(const std::string& data, char a) {
  if (data.empty()) return {};
  auto strs = str_split(data, a);
  std::vector<float> result;
  for (const auto& str_ : strs) {
    TRACE_EXCEPTION(result.push_back(std::stof(str_)));
  }
  return result;
}

std::vector<int> str2int(const std::string& data, char a) {
  if (data.empty()) return {};
  auto strs = str_split(data, a);
  std::vector<int> result;
  for (const auto& str_ : strs) {
    TRACE_EXCEPTION(result.push_back(std::stoi(str_)));
  }
  return result;
}

std::set<int> str2set(const std::string& data, char in) {
  if (data.empty()) return {};
  auto strs = str_split(data, in);
  std::set<int> result;
  for (const auto& str_ : strs) {
    TRACE_EXCEPTION(result.insert(std::stoi(str_)));
  }
  return result;
}

std::vector<std::set<int>> str2set(const std::string& data, char a, char b) {
  if (data.empty()) return {};
  std::vector<std::set<int>> results;
  auto strs = str_split(data, b);

  for (const auto& str_ : strs) {
    results.emplace_back(str2set(str_, a));
  }
  return results;
}

/// data="1x3x224x224,1x3x224x224", a='x' b=',' -> [[1,3,224,224],[1,3,224,224]]
std::vector<std::vector<int>> str2int(const std::string& data, char a, char b) {
  if (data.empty()) return {};
  std::vector<std::vector<int>> results;
  auto strs = str_split(data, b);

  for (const auto& str_ : strs) {
    results.emplace_back(str2int(str_, a));
  }
  return results;
}

std::pair<std::string, MapInfo> split_vertical_line(std::string strtem) {
  std::string::size_type pos1 = strtem.find('|');
  if (pos1 == std::string::npos || strtem.size() < 3) {
    SPDLOG_ERROR("error map config: {}", strtem);
    throw std::invalid_argument("error map config: " + strtem);
  }

  auto key = strtem.substr(0, pos1);
  auto value = strtem.substr(pos1 + 1);
  if (startswith(key, "TASK_") && endswith(key, "_KEY") &&
      TASK_KEY_MAP.find(key) != TASK_KEY_MAP.end()) {
    key = TASK_KEY_MAP.at(key);
  }
  if (startswith(value, "TASK_") && endswith(value, "_KEY") &&
      TASK_KEY_MAP.find(value) != TASK_KEY_MAP.end()) {
    value = TASK_KEY_MAP.at(value);
  }
  MapInfo tmp;
  tmp.value = value;
  tmp.map_type = MapInfo::split;
  std::swap(key, tmp.value);
  return std::pair<std::string, MapInfo>(key, tmp);
}

std::pair<std::string, MapInfo> split_double_vertical_line(std::string strtem) {
  std::string::size_type pos1 = strtem.find("||");
  if (pos1 == std::string::npos || strtem.size() < 4) {
    SPDLOG_ERROR("error map config: {}", strtem);
    throw std::invalid_argument("error map config: " + strtem);
  }

  auto key = strtem.substr(0, pos1);
  auto value = strtem.substr(pos1 + 2);
  if (startswith(key, "TASK_") && endswith(key, "_KEY") &&
      TASK_KEY_MAP.find(key) != TASK_KEY_MAP.end()) {
    key = TASK_KEY_MAP.at(key);
  }
  if (startswith(value, "TASK_") && endswith(value, "_KEY") &&
      TASK_KEY_MAP.find(value) != TASK_KEY_MAP.end()) {
    value = TASK_KEY_MAP.at(value);
  }
  MapInfo tmp;
  tmp.value = value;
  tmp.map_type = MapInfo::reduce;

  std::swap(key, tmp.value);
  return std::pair<std::string, MapInfo>(key, tmp);
}

std::pair<std::string, MapInfo> split_colon(std::string strtem) {
  std::string::size_type pos1 = strtem.find(':');
  if (pos1 == std::string::npos || strtem.size() < 2 || *strtem.begin() == ':' ||
      *(strtem.end() - 1) == ':') {
    SPDLOG_ERROR("error map config: {}", strtem);
    throw std::invalid_argument("error map config: " + strtem);
  }
  while (pos1 != std::string::npos && (strtem[pos1 + 1] == ':' || strtem[pos1 - 1] == ':')) {
    pos1 = strtem.find(':', pos1 + 1);
  }
  if (pos1 == std::string::npos) {
    SPDLOG_ERROR("error map config: {}", strtem);
    throw std::invalid_argument("error map config: " + strtem);
  } else {
    auto key = strtem.substr(0, pos1);
    auto value = strtem.substr(pos1 + 1);
    if (startswith(key, "TASK_") && endswith(key, "_KEY") &&
        TASK_KEY_MAP.find(key) != TASK_KEY_MAP.end()) {
      key = TASK_KEY_MAP.at(key);
    }
    if (startswith(value, "TASK_") && endswith(value, "_KEY") &&
        TASK_KEY_MAP.find(value) != TASK_KEY_MAP.end()) {
      value = TASK_KEY_MAP.at(value);
    }
    MapInfo tmp;
    tmp.value = value;
    std::swap(key, tmp.value);
    return std::pair<std::string, MapInfo>(key, tmp);
  }
}

std::vector<std::string> split(std::string strtem, char a) {
  std::vector<std::string> strvec;

  auto itor = std::remove(strtem.begin(), strtem.end(), ' ');
  strtem.erase(itor, strtem.end());

  std::string::size_type pos1, pos2;
  pos2 = strtem.find(a);
  pos1 = 0;
  while (std::string::npos != pos2) {
    strvec.push_back(strtem.substr(pos1, pos2 - pos1));

    pos1 = pos2 + 1;
    pos2 = strtem.find(a, pos1);
  }
  strvec.push_back(strtem.substr(pos1));
  return strvec;
}

std::unordered_map<std::string, MapInfo> split_colons(std::string strtem) {
  std::unordered_map<std::string, MapInfo> result;
  auto data = split(strtem, ',');
  for (auto item : data) {
    // replace_all(item, "|", ":");
    auto iter = item.find("||");
    if (iter != std::string::npos) {
      auto ret = result.insert(split_double_vertical_line(item));
      if (!ret.second) {
        throw std::invalid_argument("map not support: " + strtem);
      }
      continue;
    }

    iter = item.find('|');
    if (iter != std::string::npos) {
      auto ret = result.insert(split_vertical_line(item));
      if (!ret.second) {
        throw std::invalid_argument("map not support: " + strtem);
      }
    } else {
      auto ret = result.insert(split_colon(item));
      if (!ret.second) {
        throw std::invalid_argument("map not support: " + strtem);
      }
    }
  }

  return result;
}

std::pair<std::string, std::string> split_map_colon(std::string strtem) {
  std::string::size_type pos1 = strtem.find(':');
  if (pos1 == std::string::npos || strtem.size() < 2 || *strtem.begin() == ':' ||
      *(strtem.end() - 1) == ':') {
    SPDLOG_ERROR("error map config: {}", strtem);
    throw std::invalid_argument("error map config: " + strtem);
  }
  while (pos1 != std::string::npos && (strtem[pos1 + 1] == ':' || strtem[pos1 - 1] == ':')) {
    pos1 = strtem.find(':', pos1 + 1);
  }
  if (pos1 == std::string::npos) {
    SPDLOG_ERROR("error map config: {}", strtem);
    throw std::invalid_argument("error map config: " + strtem);
  } else {
    auto key = strtem.substr(0, pos1);
    auto value = strtem.substr(pos1 + 1);
    if (startswith(key, "TASK_") && endswith(key, "_KEY") &&
        TASK_KEY_MAP.find(key) != TASK_KEY_MAP.end()) {
      key = TASK_KEY_MAP.at(key);
    }
    if (startswith(value, "TASK_") && endswith(value, "_KEY") &&
        TASK_KEY_MAP.find(value) != TASK_KEY_MAP.end()) {
      value = TASK_KEY_MAP.at(value);
    }
    return std::pair<std::string, std::string>(key, value);
  }
}

std::unordered_map<std::string, std::string> split_map_colons(std::string strtem) {
  std::unordered_map<std::string, std::string> result;
  auto data = split(strtem, ',');
  for (auto item : data) result.insert(split_map_colon(item));
  return result;
}

std::unordered_map<std::string, std::unordered_map<std::string, MapInfo>> generate_map(
    std::string data) {
  auto generated_data = str_split_brackets_match(data, ',', '[', ']');
  std::unordered_map<std::string, std::unordered_map<std::string, MapInfo>> map_config;
  // static const std::regex map_regex("([\\w:-\\.])*\\[.+\\]");
  for (const auto& item : generated_data) {
    // if (!std::regex_match(item, map_regex)) {
    //   throw std::invalid_argument(item + " => Regex match failed : pattern is " +
    //                               std::string("node_name[src_key:dst_key]"));
    // }
    auto pos = item.find('[');
    auto pos_end = item.find_last_of(']');
    if (pos == std::string::npos || pos_end == std::string::npos || pos > pos_end) {
      throw std::invalid_argument(item + " => Regex match failed : pattern is " +
                                  std::string("node_name[src_key:dst_key]"));
    }

    auto key = item.substr(0, pos);
    if (map_config.find(key) != map_config.end()) {
      throw std::invalid_argument("duplicated key: " + key);
    }
    map_config[key] = split_colons(item.substr(1 + pos, pos_end - pos - 1));
  }

  return map_config;
}

std::set<std::string> get_map_reduce_nodes(const std::string& data) {
  std::set<std::string> result;

  auto config = generate_map(data);
  for (const auto& item : config) {
    for (const auto& item_inner : item.second) {
      if (item_inner.second.map_type != MapInfo::MapType::replace) {
        IPIPE_ASSERT(result.count(item_inner.first) == 0);
        result.insert(item_inner.first);
      }
    }
  }

  return result;
}

mapmap generate_mapmap(std::string data) {
  auto generated_data = str_split_brackets_match(data, ',', '[', ']');
  mapmap map_config;
  for (const auto& item : generated_data) {
    auto pos = item.find('[');
    auto pos_end = item.find_last_of(']');
    if (pos != std::string::npos) {
      auto key = item.substr(0, pos);
      map_config[key] = split_map_colons(item.substr(1 + pos, pos_end - pos - 1));
    }
  }

  return map_config;
}

bool Params::init(const std::unordered_map<std::string, std::string>& config) {
  for (auto iter = init_optinal_params_.begin(); iter != init_optinal_params_.end(); ++iter) {
    // IPIPE_ASSERT(!iter->first.empty());
    auto iter_config = config.find(iter->first);
    if (iter_config == config.end()) {
      config_[iter->first] = iter->second;
    } else {
      config_[iter->first] = iter_config->second;
    }
  }

  for (const auto& req : init_required_params_) {
    IPIPE_ASSERT(!req.empty());
    auto iter_config = config.find(req);
    if (iter_config == config.end()) {
      std::string node_name;
      auto iter_name = config.find("node_name");
      node_name = (iter_name == config.end()) ? "" : iter_name->second + ": ";
      SPDLOG_ERROR(node_name + "param not set : " + req);
      return false;
    } else {
      config_[req] = iter_config->second;
    }
  }

  for (const auto& req : init_or_forward_required_params_) {
    auto iter_config = config.find(req);
    if (iter_config != config.end()) {
      config_[req] = iter_config->second;
    }
  }
  return true;
}

void Params::check(const std::unordered_map<std::string, std::string>& config) {
  for (auto iter = init_optinal_params_.begin(); iter != init_optinal_params_.end(); ++iter) {
    // IPIPE_ASSERT(!iter->first.empty());
    auto iter_config = config.find(iter->first);
    if (iter_config == config.end()) {
      config_[iter->first] = iter->second;
    } else {
      config_[iter->first] = iter_config->second;
    }
  }

  for (const auto& req : init_required_params_) {
    IPIPE_ASSERT(!req.empty());
    auto iter_config = config.find(req);
    if (iter_config == config.end()) {
      std::string node_name;
      auto iter_name = config.find("node_name");
      node_name = (iter_name == config.end()) ? "" : iter_name->second + ": ";
      throw std::invalid_argument(node_name + "param not set : " + req);

    } else {
      config_[req] = iter_config->second;
    }
  }

  for (const auto& req : init_or_forward_required_params_) {
    auto iter_config = config.find(req);
    if (iter_config != config.end()) {
      config_[req] = iter_config->second;
    }
  }
}

void Params::check_and_update(dict forward_data) {
  for (auto iter = forward_optinal_params_.begin(); iter != forward_optinal_params_.end(); ++iter) {
    auto iter_config = forward_data->find(iter->first);
    if (iter_config == forward_data->end()) {
      if (config_.find(iter->first) == config_.end()) config_[iter->first] = iter->second;
    } else {
      config_[iter->first] = any_cast<std::string>(iter_config->second);
    }
  }

  for (const auto& req : forward_required_params_) {
    auto iter_config = forward_data->find(req);
    if (iter_config == forward_data->end()) {
      SPDLOG_ERROR("[update Params] not set : " + req);
      throw std::invalid_argument("[update Params] not set : " + req);
    } else {
      config_[req] = any_cast<std::string>(iter_config->second);
    }
  }

  for (const auto& req : init_or_forward_required_params_) {
    auto iter_config = forward_data->find(req);
    if (iter_config == forward_data->end()) {
      if (config_.find(req) == config_.end()) {
        SPDLOG_ERROR("[update Params] not set : " + req);
        throw std::invalid_argument("[update Params] not set : " + req);
      }
    } else {
      config_[req] = any_cast<std::string>(iter_config->second);
    }
  }
  return;
}

/**
 * @brief 判断字符串左右括号是否匹配，是否能被外层的逗号分开；
 *
 * @param strtem
 * @param left
 * @param right
 * @return true   例如 B[C],D,B[E[Z1,Z2]] 可被分为三个部分
 * @return false  例如 A[B[C],D,B[E[Z1,Z2]]] 无法被逗号分开
 * @exception
 * 如果括号不匹配，逗号出现在字符串第一个或者最后一个位置会抛出invalid_argument异常
 */
bool is_comma_separable(const std::string& strtem, char left, char right) {
  auto index_left = strtem.find(',');

  // 逗号位置异常
  if (index_left == 0 || index_left == strtem.size() - 1) {
    SPDLOG_ERROR("location of ',' weired: " + strtem);
    throw std::invalid_argument("location of ',' weired: " + strtem);
  }

  /// 只有一种括号类型，通过模拟栈的大小匹配左右括号
  std::size_t stack_size = 0;
  bool separable = false;
  for (const auto& item : strtem) {
    if (item == left) {
      stack_size++;
    } else if (item == right) {
      if (stack_size == 0) {
        SPDLOG_ERROR("brackets not match: " + strtem);
        throw std::invalid_argument("brackets not match: " + strtem);
      }
      stack_size--;
    } else if (item == ',') {
      // 出现逗号且左边的括号已经匹配完毕的话，代表字符串可以从此逗号分开
      if (stack_size == 0) {
        separable = true;
      }
    }
  }
  if (stack_size != 0) {
    SPDLOG_ERROR("brackets not match " + strtem);
    throw std::invalid_argument("brackets not match " + strtem);
  }
  // 没有逗号则不可分
  if (index_left == std::string::npos) {
    return false;
  }
  return separable;
}
std::vector<std::string> brackets_match(const std::string& strtem_, char left, char right) {
  std::vector<std::string> brackets;
  std::string strtem = strtem_;
  auto itor = std::remove(strtem.begin(), strtem.end(), ' ');
  strtem.erase(itor, strtem.end());

  while (!strtem.empty()) {
    if (brackets.size() > 10000) throw std::invalid_argument("too many []");
    auto iter_begin = strtem.find(left);
    if (iter_begin != std::string::npos) {
      auto iter_end = strtem.find_last_of(right);

      if (iter_end == std::string::npos) {
        throw std::invalid_argument("brackets not match: " + strtem_);
      }
      assert(iter_end == strtem.size() - 1);

      brackets.emplace_back(strtem.substr(0, iter_begin));
      strtem = strtem.substr(iter_begin + 1, iter_end - iter_begin - 1);

      // 如果strtem是用逗号分开的多个Backend，则作为一个整体不解析
      if (is_comma_separable(strtem, left, right)) {
        brackets.push_back(strtem);
        break;
      }

    } else {
      brackets.push_back(strtem);
      break;
    }
  }
  return brackets;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
void brackets_split(std::unordered_map<std::string, std::string>& config, std::string key,
                    char left, char right) {
  auto iter = config.find(key);
  if (iter == config.end()) {
    return;
  } else {
    brackets_split(iter->second, config, key, left, right);
  }
}
#endif

std::string pre_parentheses_split(const std::string& strtem, std::string& pre_str) {
  constexpr auto left = '(';
  constexpr auto right = ')';
  auto iter = strtem.find(left);
  auto iter_right = strtem.find_last_of(right, strtem.find('['));
  if (iter != 0 || iter_right == std::string::npos) {
    return strtem;
  } else {
    pre_str = strtem.substr(iter + 1, iter_right - 1);
    return strtem.substr(iter_right + 1);
  }
}

std::string post_parentheses_split(const std::string& strtem, std::string& post) {
  constexpr auto left = '(';
  constexpr auto right = ')';
  auto iter = strtem.find(left);
  auto iter_right = strtem.find_last_of(right, strtem.find('['));

  if (iter == std::string::npos) {
    IPIPE_ASSERT(iter_right == std::string::npos);
    return strtem;
  } else {
    IPIPE_ASSERT(iter != 0);
    IPIPE_ASSERT(iter_right != std::string::npos);
    post = strtem.substr(iter + 1, iter_right - 1 - iter);
    return strtem.substr(0, iter) + strtem.substr(iter_right + 1);
  }
}

void brackets_split(const std::string& strtem_,
                    std::unordered_map<std::string, std::string>& config, std::string key,
                    char left, char right) {
  auto brackets = brackets_match(strtem_, left, right);

  if (brackets.empty()) {
    SPDLOG_ERROR("error backend: " + strtem_);
    throw std::invalid_argument("error backend: " + strtem_);
  }
  std::unordered_map<std::string, std::string> new_config;
  new_config[key] = brackets[0];
  for (std::size_t i = 1; i < brackets.size(); ++i) {
    auto iter = new_config.find(brackets[i - 1] + "::backend");
    if (iter != new_config.end()) {
      SPDLOG_ERROR("Recursive backend({}) is not allowed. backend={}", brackets[i - 1], strtem_);

      throw std::invalid_argument("Recursive backend(" + brackets[i - 1] +
                                  ") is not allowed. Backend is " + strtem_);
    }
    new_config[brackets[i - 1] + "::backend"] = brackets[i];
  }
  for (auto iter = new_config.begin(); iter != new_config.end(); ++iter) {
    config[iter->first] = iter->second;
  }
}

std::string brackets_combine(const std::unordered_map<std::string, std::string>& config) {
  auto iter = config.find("backend");
  if (iter == config.end()) throw std::invalid_argument("backend not found ");
  std::string back_end_name = iter->second;
  std::string back_end_name_iter = back_end_name;
  std::string tial = "";
  while (config.find(back_end_name_iter + "::backend") != config.end()) {
    back_end_name_iter = config.at(back_end_name_iter + "::backend");
    back_end_name += "[" + back_end_name_iter;
    tial += "]";
  }
  back_end_name += tial;
  return back_end_name;
}

}  // namespace ipipe