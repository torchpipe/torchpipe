#include <queue>
#include <stdexcept>
// #include <algorithm>
#include <stack>

#include "omniback/core/backend.hpp"
#include "omniback/core/parser.hpp"
#include "omniback/core/task_keys.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
#include "omniback/helper/string.hpp"
#include <tvm/ffi/extra/stl.h>

namespace omniback::parser {

void broadcast_global(str::mapmap& config) {
  auto iter = config.find(TASK_GLOBAL_KEY);
  if (iter == config.end())
    return;

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
              cited_item + " has no map config, default set to [result:data]");
          dag_config_[cited_item].map_config = str::mapmap();
          dag_config_[cited_item].map_config[item.first] =
              str::str_map{{TASK_DATA_KEY, TASK_RESULT_KEY}};
        }
      }
    }
  }

  // check or
  for (const auto& item : dag_config_) {
    if (item.second.cited.size() > 1) {
      OMNI_ASSERT(
          !item.second.or_filter,
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
      OMNI_ASSERT(
          iter != dag_config_.end(), "DagParser: next node not found: " + next);
      // item.first => next
      iter->second.previous.insert(item.first);
    }
    if (item.second.map_config.empty()) {
      OMNI_ASSERT(
          item.second.previous.size() <= 1,
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
  OMNI_ASSERT(!roots_.empty());
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
    OMNI_ASSERT(
        maps_data.size() == 2,
        "map should be in the form of node_name[src_key:dst_key]");
    in_map = str::map_split(maps_data[1], ':', ',', "", true);
    if (!in_map.empty()) {
      map_config[maps_data[0]] = in_map;
    }
  }
  return map_config;
}

std::unordered_set<std::string> DagParser::get_subgraph(
    const std::string& root) {
  // std::unordered_set<std::string> re;
  OMNI_ASSERT(
      dag_config_.find(root) != dag_config_.end(),
      "DagParser: " + root + " not found");
  // OMNI_ASSERT(dag_config_[root].previous.empty(), root + " is not a root
  // node");

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
  // for (const auto& item : parsered) {
  //   for (const auto& pre : dag_config_[item].previous) {
  //     OMNI_ASSERT(
  //         parsered.find(pre) != parsered.end(),
  //         "DagParser: " + pre + " is not in subgraph");
  //   }
  // }
  return parsered;
}

dict DagParser::prepare_data_from_previous(
    const std::string& node,
    std::unordered_map<std::string, dict>& processed) {
  if (dag_config_.at(node).map_config.empty()) {
    auto previous_node = *dag_config_.at(node).previous.begin();
    auto re = processed.at(previous_node);
    if (re->find(TASK_RESULT_KEY) == re->end()) {
      if (!dag_config_.at(node).or_filter) {
        throw std::runtime_error(
            "DagParser: " + previous_node + " has no result");
      }
    } else {
      if (dag_config_.at(node).or_filter) {
        processed[node] = re;
        return re;
      } else {
        (*re)[TASK_DATA_KEY] = re->at(TASK_RESULT_KEY);
        re->erase(TASK_RESULT_KEY);
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
    for (const auto& [dst_key, src_key] : item.second) {
      auto iter = src_dict->find(src_key);
      OMNI_ASSERT(
          iter != src_dict->end(),
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

  OMNI_ASSERT(
      re->find(TASK_DATA_KEY) != re->end(),
      std::string("DagParser: ") + TASK_DATA_KEY + " not found in target");
  return re;
}

void update(
    const std::unordered_map<std::string, std::string>& config,
    std::unordered_map<std::string, std::string>& str_kwargs) {
  for (const auto& [key, value] : config) {
    if (str_kwargs.find(key) == str_kwargs.end()) {
      str_kwargs[key] = value;
    }
  }
}

// Split the string by the delimiter, ignoring delimiters inside nested brackets
// (including [], {}, ())
std::vector<std::string> split_args(const std::string& s, char delimiter) {
  std::vector<std::string> args;
  std::stack<char> brackets;
  size_t start = 0;

  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    // Handle bracket matching
    if (c == '[' || c == '{' || c == '(') {
      brackets.push(c);
    } else if (c == ']') {
      if (brackets.empty() || brackets.top() != '[')
        throw std::runtime_error("Mismatched ']'");
      brackets.pop();
    } else if (c == '}') {
      if (brackets.empty() || brackets.top() != '{')
        throw std::runtime_error("Mismatched '}'");
      brackets.pop();
    } else if (c == ')') {
      if (brackets.empty() || brackets.top() != '(')
        throw std::runtime_error("Mismatched ')'");
      brackets.pop();
    }

    // Split only when not inside brackets
    if (c == delimiter && brackets.empty()) {
      // Add non-empty segments only
      if (i > start) {
        args.push_back(s.substr(start, i - start));
      }
      start = i + 1;
    }
  }

  // Add last segment if non-empty
  if (start < s.size()) {
    args.push_back(s.substr(start));
  }

  // Validate bracket balance
  if (!brackets.empty())
    throw std::runtime_error("Mismatched brackets");

  return args;
}

// Parse individual argument into key-value pair if valid
std::pair<bool, std::pair<std::string, std::string>> parse_kwarg(
    const std::string& arg,
    char delimiter) {
  std::stack<char> brackets;
  size_t eq_pos = std::string::npos;
  std::string delim_str(1, delimiter);

  for (size_t i = 0; i < arg.size(); ++i) {
    char c = arg[i];
    // Track bracket nesting
    if (c == '[' || c == '{' || c == '(') {
      brackets.push(c);
    } else if (c == ']' || c == '}' || c == ')') {
      if (brackets.empty()) {
        std::stringstream ss;
        ss << "Mismatched closing bracket '" << c << "'";
        throw std::runtime_error(ss.str());
      }
      char top = brackets.top();
      if ((c == ']' && top != '[') || (c == '}' && top != '{') ||
          (c == ')' && top != '(')) {
        std::stringstream ss;
        ss << "Mismatched closing bracket '" << c << "' for opening '" << top
           << "'";
        throw std::runtime_error(ss.str());
      }
      brackets.pop();
    }

    // Find delimiter outside brackets
    if (brackets.empty() && c == delimiter) {
      if (eq_pos != std::string::npos) {
        std::stringstream ss;
        ss << "Multiple '" << delimiter << "' in keyword arg";
        throw std::runtime_error(ss.str());
      }
      eq_pos = i;
    }
  }

  // Validate delimiter position
  if (eq_pos == std::string::npos)
    return {false, {}};

  if (eq_pos == 0) {
    std::stringstream ss;
    ss << "Empty key before '" << delimiter << "'";
    throw std::runtime_error(ss.str());
  }

  if (eq_pos == arg.size() - 1) {
    std::stringstream ss;
    ss << "Empty value after '" << delimiter << "'";
    throw std::runtime_error(ss.str());
  }

  return {true, {arg.substr(0, eq_pos), arg.substr(eq_pos + 1)}};
}

// Main parsing function with Python-style argument rules
// Parse a configuration string into positional arguments and keyword arguments
std::
    pair<std::vector<std::string>, std::unordered_map<std::string, std::string>>
    parse_args_kwargs(std::string config) {
  // Remove whitespace and control characters
  config.erase(
      std::remove_if(
          config.begin(),
          config.end(),
          [](char c) { return std::isspace(c) || std::iscntrl(c); }),
      config.end());

  // Handle empty input
  if (config.empty()) {
    return {};
  }

  auto args = split_args(config, ',');
  std::pair<
      std::vector<std::string>,
      std::unordered_map<std::string, std::string>>
      result;
  bool kwarg_started = false;

  for (const auto& arg : args) {
    if (arg.empty())
      continue;

    try {
      auto [is_kwarg, kv] = parse_kwarg(arg, '=');
      if (is_kwarg) {
        kwarg_started = true;
        // Check for duplicate keys
        if (result.second.count(kv.first)) {
          std::stringstream ss;
          ss << "Duplicate keyword argument: " << kv.first;
          throw std::runtime_error(ss.str());
        }
        result.second.insert(kv);
      } else {
        // Validate argument order
        if (kwarg_started) {
          throw std::runtime_error(
              "Positional argument after keyword argument");
        }
        result.first.push_back(arg);
      }
    } catch (const std::runtime_error& e) {
      // Enhance error context
      std::stringstream ss;
      ss << "Invalid argument '" << arg << "': " << e.what();
      throw std::runtime_error(ss.str());
    }
  }

  return result;
}
} // namespace omniback::parser

namespace omniback::parser_v2 {
bool has_valid_unnested_delimiters(
    const std::string& input,
    const std::vector<BracketPair>& bracket_pairs,
    const std::unordered_set<char>& delimiters) {
  // Precondition validation
  if (!delimiters.empty() && !input.empty()) {
    const char first = input.front();
    const char last = input.back();
    if (delimiters.count(first)) {
      SPDLOG_ERROR("Invalid leading delimiter '{}' in: {}", first, input);
      throw std::invalid_argument("Leading delimiter");
    }
    if (delimiters.count(last)) {
      SPDLOG_ERROR("Invalid trailing delimiter '{}' in: {}", last, input);
      throw std::invalid_argument("Trailing delimiter");
    }
  }

  // Build lookup structures
  std::unordered_map<char, char> left_to_right;
  std::unordered_map<char, char> right_to_left;
  std::unordered_set<char> valid_left_brackets;
  std::unordered_set<char> valid_right_brackets;

  for (const auto& pair : bracket_pairs) {
    if (left_to_right.count(pair.left) || right_to_left.count(pair.right)) {
      SPDLOG_ERROR(
          "Duplicate bracket definition: {}-{}", pair.left, pair.right);
      throw std::invalid_argument("Duplicate bracket pair");
    }

    left_to_right[pair.left] = pair.right;
    right_to_left[pair.right] = pair.left;
    valid_left_brackets.insert(pair.left);
    valid_right_brackets.insert(pair.right);
  }

  std::stack<char> bracket_stack;
  bool found_unnested_delimiter = false;

  // Main parsing loop
  for (size_t pos = 0; pos < input.size(); ++pos) {
    const char c = input[pos];

    if (valid_left_brackets.count(c)) {
      bracket_stack.push(c);
    } else if (valid_right_brackets.count(c)) {
      if (bracket_stack.empty() || left_to_right[bracket_stack.top()] != c) {
        SPDLOG_ERROR("Mismatched '{}' at position {} in: {}", c, pos, input);
        throw std::invalid_argument("Bracket mismatch");
      }
      bracket_stack.pop();
    } else if (delimiters.count(c)) {
      // Validate delimiter context
      if (pos == 0 || pos == input.size() - 1) { // Redundant check for safety
        SPDLOG_ERROR(
            "Invalid delimiter '{}' at string boundary in: {}", c, input);
        throw std::invalid_argument("Boundary delimiter");
      }

      // Check if delimiter is outside all brackets
      if (bracket_stack.empty()) {
        found_unnested_delimiter = true;
      }
    }
  }

  // Post-validation
  if (!bracket_stack.empty()) {
    SPDLOG_ERROR("Unclosed '{}' bracket in: {}", bracket_stack.top(), input);
    throw std::invalid_argument("Unclosed bracket");
  }

  return found_unnested_delimiter;
}

void remove_space_and_ctrl(std::string& strtem) {
  strtem.erase(
      std::remove_if(
          strtem.begin(),
          strtem.end(),
          [](unsigned char c) { return std::isspace(c) || std::iscntrl(c); }),
      strtem.end());
}

/**
 * @brief Determines if an input string can be split by delimiters outside
 * bracket-enclosed regions.
 *
 * This function splits the input string into tokens separated by specified
 * delimiters, while treating content within bracket pairs as indivisible
 * blocks. Bracket nesting is fully supported. The function populates two
 * vectors: one containing the extracted delimiters in order, and another
 * containing the resulting tokens.
 *
 * @param input The string to be processed. Expected to have balanced brackets
 * and spaces/control characters removed(not verified internally).
 * @param bracket_pairs Bracket pairs to consider as block boundaries. Default:
 * { '()', '{}', '[]' }.
 * @param delimiters Characters to use as token separators outside brackets.
 * Default: { ',', ';' }.
 * @param[out] result_delimiters Receives encountered delimiters in order. Size
 * will be tokens.size()-1.
 * @param[out] result_output Receives parsed tokens with
 * result_delimiters.size() + 1 == result_output.size() always hold.
 * @return true If at least one valid delimiter was found and splitting
 * occurred.
 * @return false If no delimiters were found in non-bracketed regions.
 *
 * @note
 * - Empty tokens may result from consecutive delimiters or leading/trailing
 * delimiters
 * - Delimiters inside any bracket type are always ignored
 *
 * Example:
 * @code
 * Input: "a(b,c); d", delimiters {',', ';'}
 * Output: tokens ["ab", "d"], delimiters [';']
 * @endcode
 */
bool is_delimiter_separable(
    const std::string& input,
    std::vector<char>& result_delimiters,
    std::vector<std::string>& result_output,
    const std::unordered_map<char, char>& left_to_right =
        {{'(', ')'}, {'{', '}'}, {'[', ']'}},
    const std::unordered_set<char>& delimiters = {',', ';'}) {
  result_delimiters.clear();
  result_output.clear();

  // Create a map from left brackets to their corresponding right brackets

  std::stack<char> bracket_stack;
  size_t start_pos = 0;

  for (size_t i = 0; i < input.size(); ++i) {
    const char c = input[i];
    auto it = left_to_right.find(c);
    if (it != left_to_right.end()) {
      // Push the corresponding right bracket onto the stack
      bracket_stack.push(it->second);
    } else if (!bracket_stack.empty() && c == bracket_stack.top()) {
      // Matching right bracket found, pop the stack
      bracket_stack.pop();
    } else if (delimiters.count(c) && bracket_stack.empty()) {
      // Found a delimiter outside any brackets, split here
      std::string token = input.substr(start_pos, i - start_pos);
      result_output.push_back(token);
      result_delimiters.push_back(c);
      start_pos = i + 1;
    }
  }

  // Process the last token after the last delimiter
  std::string last_token = input.substr(start_pos);
  result_output.push_back(last_token);

  OMNI_FATAL_ASSERT(
      result_delimiters.size() + 1 == result_output.size(), "size not match");
  // Return true if at least one delimiter was found and processed
  return !result_delimiters.empty();
}

bool is_delimiter_separable(
    const std::string& input,
    const std::unordered_map<char, char>& left_to_right,
    const std::unordered_set<char>& delimiters) {
  OMNI_ASSERT(!input.empty());
  if (left_to_right.count(input[0])) {
    return false;
  }

  std::stack<char> bracket_stack;

  for (size_t i = 0; i < input.size(); ++i) {
    const char c = input[i];
    auto it = left_to_right.find(c);
    if (it != left_to_right.end()) {
      // Push the corresponding right bracket onto the stack
      bracket_stack.push(it->second);
    } else if (!bracket_stack.empty() && c == bracket_stack.top()) {
      // Matching right bracket found, pop the stack
      bracket_stack.pop();
    } else if (delimiters.count(c) && bracket_stack.empty()) {
      // Found a delimiter outside any brackets, split here
      return true;
    }
  }

  return false;
}

/**
 * @brief Verifies if all brackets in a string are properly nested and balanced.
 *
 * @param input The string to validate for bracket balance
 * @param bracket_pairs Bracket pairs to consider during validation. Default:
 *        { '()', '{}', '[]' }
 * @return true If all brackets are properly balanced and nested
 * @return false If any bracket mismatch or imbalance is detected
 *
 * @par Example:
 * @code
 * are_brackets_balanced("([{}])") // returns true
 * are_brackets_balanced("({[}])") // returns false (mismatched order)
 * are_brackets_balanced("a(b)c[d]e{f}") // returns true
 * @endcode
 */
bool are_brackets_balanced(
    const std::string& input,
    const std::unordered_map<char, char> open_to_close = {
        {'(', ')'},
        {'{', '}'},
        {'[', ']'}}) {
  // Create fast lookup structures
  std::unordered_set<char> close_brackets;

  // Prevent duplicate bracket definitions
  std::unordered_set<char> seen;

  for (const auto& pair : open_to_close) {
    // Validate bracket pair uniqueness
    if (seen.count(pair.first) || seen.count(pair.second)) {
      throw std::invalid_argument(
          "Duplicate bracket characters in pair definitions");
    }

    close_brackets.insert(pair.second);
    seen.insert(pair.first);
    seen.insert(pair.second);
  }

  std::stack<char> bracket_stack;

  for (char c : input) {
    if (open_to_close.count(c)) {
      // Found opening bracket - push expected closing bracket
      bracket_stack.push(open_to_close.at(c));
    } else if (close_brackets.count(c)) {
      // Found closing bracket - check stack integrity
      if (bracket_stack.empty() || bracket_stack.top() != c) {
        return false;
      }
      bracket_stack.pop();
    }
    // Ignore non-bracket characters
  }

  // All brackets must be closed by end of input
  return bracket_stack.empty();
}

std::vector<std::pair<std::string, char>> flatten_brackets(
    const std::string& strtem_in) {
  std::vector<std::pair<std::string, char>> re =
      expend_outmost_brackets(strtem_in);
  OMNI_ASSERT(
      re.size() >= 1 && re.size() <= 3 && re[0].second == 0,
      "illegal input: " + strtem_in + ". Support A(args, kwsrag)[B]");

  if (re.size() == 1) {
    return {{strtem_in, 0}};
  }
  if (re.size() == 2) {
    OMNI_ASSERT(
        re[1].second == '(' || re[1].second == '[',
        "illegal input: " + strtem_in);
  }
  if (re.size() == 3) {
    OMNI_ASSERT(
        re[1].second == '(' && re[2].second == '[',
        "illegal input: " + strtem_in);
  }
  if (re.back().second == '[') {
    OMNI_ASSERT(!re.back().first.empty());
    if (is_delimiter_separable((re.back().first)) ||
        re.back().first[0] == '(') {
      return re;
    }
    auto new_re = flatten_brackets(re.back().first);
    re.back().first = new_re.front().first;
    for (auto iter = new_re.begin() + 1; iter != new_re.end(); ++iter) {
      re.push_back(*iter);
    }
  }
  return re;
}

std::string remove_bracket(const std::string& input, char bracket) {
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] == bracket) {
      return input.substr(i + 1, input.size() - i - 1);
    }
  }
  throw std::invalid_argument("Bracket not found");
  return "";
}

std::pair<std::string, std::string> Parser::prifix_split(
    const std::string& input,
    char left_bracket,
    char right_bracket) {
  OMNI_ASSERT(!input.empty());
  if (input[0] != left_bracket) {
    return {"", input};
  }
  std::stack<char> bracket_stack;
  bracket_stack.push(left_bracket);
  std::pair<std::string, std::string> result;
  for (size_t i = 1; i < input.size(); ++i) {
    if (input[i] == left_bracket) {
      bracket_stack.push(input[i]);
    } else if (input[i] == right_bracket) {
      if (bracket_stack.empty()) {
        throw std::invalid_argument("Mismatched brackets");
      } else if (bracket_stack.size() == 1) {
        result = {
            input.substr(1, i - 1), input.substr(i + 1, input.size() - i - 1)};
        return result;
      }
      bracket_stack.pop();
    }
  }
  throw std::invalid_argument("Mismatched brackets");
  return {};
}

std::string Parser::parse(
    const std::string& params,
    std::unordered_map<std::string, std::string>& config_output) {
  auto input = params;
  remove_space_and_ctrl(input);
  std::vector<std::pair<std::string, char>> result = flatten_brackets(input);
  OMNI_ASSERT(!result.empty());
  std::unordered_set<size_t> args_index;

  std::unordered_set<std::string> keys;
  for (size_t i = 1; i < result.size(); i++) {
    if (result[i].second == '(') {
      auto dep = result[i - 1].first;
      OMNI_ASSERT(!dep.empty());
      // if (dep[0] == '(')
      // {
      //     dep = remove_bracket(dep, ')');
      // }

      const std::string& key_name = dep + "::args";
      OMNI_ASSERT(keys.count(key_name) == 0, "Duplicate key: " + key_name);
      keys.insert(key_name);

      // auto insert_re =
      config_output.insert_or_assign(key_name, result[i].first);
      // OMNI_ASSERT(insert_re.second,
      //             "Duplicate args key: " + dep +
      //                 "::args"); // result[i - 1].second == 0 &&
      args_index.insert(i);
    }
  }

  std::vector<std::pair<std::string, char>> new_brackets;
  for (size_t i = 0; i < result.size(); i++) {
    if (args_index.find(i) == args_index.end()) {
      new_brackets.push_back(result[i]);
    }
  }

  for (size_t i = 1; i < new_brackets.size(); i++) {
    const std::string& key_name = new_brackets[i - 1].first + "::dependency";
    OMNI_ASSERT(keys.count(key_name) == 0, "Duplicate key: " + key_name);
    keys.insert(key_name);
    config_output.insert_or_assign(key_name, new_brackets[i].first);
    // auto insert_re =
    //     config_output.insert({new_brackets[i - 1].first + "::dependency",
    //                           new_brackets[i].first});
    // OMNI_ASSERT(insert_re.second,
    //             "Duplicate dependency key: " + new_brackets[i - 1].first +
    //                 "::dependency");
  }

  return result[0].first;
}

std::vector<std::string> Parser::split_by_delimiter(
    const std::string& input,
    char delimiter) {
  std::vector<char> result_delimiters;
  std::vector<std::string> result_output;
  if (!is_delimiter_separable(
          input,
          result_delimiters,
          result_output,
          {{'(', ')'}, {'{', '}'}, {'[', ']'}},
          {delimiter})) {
    return {input};
  };
  return result_output;
}

std::pair<std::vector<char>, std::vector<std::string>> Parser::
    split_by_delimiters(
        const std::string& input,
        char delimiter,
        char delimiter_outter) {
  std::vector<char> result_delimiters;
  std::vector<std::string> result_output;
  if (!is_delimiter_separable(
          input,
          result_delimiters,
          result_output,
          {{'(', ')'}, {'{', '}'}, {'[', ']'}},
          {delimiter, delimiter_outter})) {
    return {{}, {input}};
  };
  return {result_delimiters, result_output};
}

std::vector<std::pair<std::string, char>> expend_outmost_brackets(
    const std::string& input,
    std::unordered_map<char, char> open_to_close_bracket) {
  std::unordered_set<char> valid_right_brackets;
  for (const auto& pair : open_to_close_bracket) {
    valid_right_brackets.insert(pair.second);
  }

  // std::map<size_t, std::pair<size_t, char>> open_to_close_bracket_location;

  std::stack<char> bracket_stack;

  std::vector<size_t> bracket_pos;
  std::vector<char> bracket_type;

  for (size_t i = 0; i < input.size();) {
    const char c = input[i];

    if (valid_right_brackets.count(c)) {
      OMNI_ASSERT(
          !bracket_stack.empty() &&
              open_to_close_bracket[bracket_stack.top()] == c,
          "Mismatched brackets.");
      if (bracket_stack.size() == 1) {
        bracket_pos.push_back(i);
        // bracket_type.push_back(bracket_stack.top());
      }
      bracket_stack.pop();
    } else if (open_to_close_bracket.count(c)) {
      if (bracket_stack.empty()) {
        bracket_pos.push_back(i);
        bracket_type.push_back(c);
      }
      bracket_stack.push(c);
    } else if (bracket_stack.empty()) {
      bracket_pos.push_back(i);
      bracket_type.push_back(0);
      int j = i + 1;
      for (; j < input.size(); ++j) {
        if (open_to_close_bracket.count(input[j])) {
          i = j;
          break;
        }
      }
      if (i == j) {
        bracket_pos.push_back(j - 1);
        continue;
      } else {
        bracket_pos.push_back(j);
        break;
      }
    }
    ++i;
  }
  OMNI_ASSERT(
      bracket_pos.size() == 2 * bracket_type.size() && !bracket_pos.empty());
  std::vector<std::pair<std::string, char>> result;

  for (size_t i = 0; i < bracket_type.size(); ++i) {
    if (bracket_type[i] == 0)
      result.emplace_back(
          input.substr(
              bracket_pos[2 * i],
              bracket_pos[2 * i + 1] - bracket_pos[2 * i] + 1),
          bracket_type[i]);
    else {
      result.emplace_back(
          input.substr(
              bracket_pos[2 * i] + 1,
              bracket_pos[2 * i + 1] - bracket_pos[2 * i] - 1),
          bracket_type[i]);
    }
  }

  // prefix `()`
  OMNI_ASSERT(!result.empty());
  if (result[0].second != 0) {
    OMNI_ASSERT(
        result.size() > 0 && result[0].second == '(' && result[1].second == 0);
    result[1].first = result[0].second + result[0].first +
        open_to_close_bracket[result[0].second] + result[1].first;
    result.erase(result.begin());
  }

  return result;
}
} // namespace omniback::parser_v2

namespace omniback {
class ParserTest : public BackendOne {
  void forward(const dict& data) override {
    std::string config = dict_get<std::string>(data, TASK_DATA_KEY);
    OMNI_ASSERT(!config.empty());
    omniback::parser_v2::Parser parser;

    std::unordered_map<std::string, std::string> config_output;
    if (omniback::parser_v2::is_delimiter_separable(config) ||
        config[0] == '(') {
      auto direct_split = parser.split_by_delimiter(config);
      std::vector<std::string> result;
      for (const auto& item : direct_split) {
        if (item[0] == '(') {
          auto one_re = parser.prifix_split(item, '(', ')');
          result.push_back(one_re.first);
          result.push_back(one_re.second);
        } else {
          result.push_back("");
          result.push_back(item);
        }
      }
      data->insert_or_assign(TASK_RESULT_KEY, result);
    } else {
      std::string main_bkd = parser.parse(config, config_output);

      data->insert_or_assign(
          TASK_RESULT_KEY, std::make_tuple(main_bkd, config_output));
    }
  }
};
OMNI_REGISTER_BACKEND(ParserTest, "ParserTest");
} // namespace omniback
