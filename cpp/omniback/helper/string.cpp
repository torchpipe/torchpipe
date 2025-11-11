#include "omniback/helper/string.hpp"
#include <algorithm>
#include <cctype> //   isspace   iscntrl
#include <stack>
#include <string>
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"

namespace omniback::str {

std::vector<std::string> str_split(std::string strtem, char a) {
  remove_space_and_ctrl(strtem);
  if (strtem.empty())
    return {};

  std::vector<std::string> strvec;

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

std::vector<std::string> items_split(
    std::string strtem,
    char a,
    char left,
    char right) {
  std::vector<std::string> strvec;
  if (strtem.empty())
    return strvec;

  remove_space_and_ctrl(strtem);

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

bool is_comma_semicolon_separable(
    const std::string& strtem,
    char left,
    char right) {
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

void remove_space_and_ctrl(std::string& strtem) {
  strtem.erase(
      std::remove_if(
          strtem.begin(),
          strtem.end(),
          [](unsigned char c) { return std::isspace(c) || std::iscntrl(c); }),
      strtem.end());
}

std::vector<std::string> flatten_brackets(
    const std::string& strtem_in,
    char left,
    char right) {
  std::vector<std::string> brackets;
  std::string strtem = strtem_in;
  remove_space_and_ctrl(strtem);

  while (!strtem.empty()) {
    if (brackets.size() > 10000)
      throw std::invalid_argument("too many []");
    auto iter_begin = strtem.find(left);
    if (iter_begin != std::string::npos) {
      auto iter_end = strtem.find_last_of(right);

      if (iter_end == std::string::npos) {
        throw std::invalid_argument("brackets not match: " + strtem_in);
      }
      OMNI_ASSERT(
          iter_end == strtem.size() - 1, "brackets not match: " + strtem_in);

      brackets.emplace_back(strtem.substr(0, iter_begin));
      strtem = strtem.substr(iter_begin + 1, iter_end - iter_begin - 1);

      // 如果strtem是用逗号分开的多个Backend，则作为一个整体不解析
      if (is_comma_semicolon_separable(strtem, left, right)) {
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

void brackets_split(
    const std::string& strtem_,
    std::unordered_map<std::string, std::string>& config,
    std::string key,
    char left,
    char right) {
  auto re = brackets_split(strtem_, config, left, right);
  config[key] = re;
}

std::string brackets_split(
    const std::string& strtem_,
    std::unordered_map<std::string, std::string>& config,
    char left,
    char right) {
  // SPDLOG_INFO("brackets_split: " + strtem_);
  auto brackets = flatten_brackets(strtem_, left, right);

  if (brackets.empty()) {
    SPDLOG_ERROR("error backend: " + strtem_);
    throw std::invalid_argument("error backend: " + strtem_);
  }
  std::unordered_map<std::string, std::string> new_config;
  // new_config[key] = brackets[0];
  for (std::size_t i = 1; i < brackets.size(); ++i) {
    auto iter = new_config.find(brackets[i - 1] + "::dependency");
    if (iter != new_config.end()) {
      SPDLOG_ERROR(
          "Recursive backend({}) is not allowed. backend={}",
          brackets[i - 1],
          strtem_);

      throw std::invalid_argument(
          "Recursive backend(" + brackets[i - 1] +
          ") is not allowed. Backend is " + strtem_);
    }
    new_config[brackets[i - 1] + "::dependency"] = brackets[i];
  }
  for (auto iter = new_config.begin(); iter != new_config.end(); ++iter) {
    config[iter->first] = iter->second;
  }
  return brackets[0];
}

std::string prefix_parentheses_split(
    const std::string& strtem,
    std::string& pre_str) {
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

std::string post_parentheses_split(
    const std::string& strtem,
    std::string& post) {
  constexpr auto left = '(';
  constexpr auto right = ')';
  auto iter = strtem.find(left);
  auto iter_right = strtem.find_last_of(right, strtem.find('['));

  if (iter > strtem.find('[')) { // A[(b)B]
    return strtem;
  }

  if (iter == std::string::npos) {
    OMNI_ASSERT(iter_right == std::string::npos, "error. input = " + strtem);
    return strtem;
  } else {
    OMNI_ASSERT(iter != 0);
    OMNI_ASSERT(
        iter_right != std::string::npos, "strtem=" + strtem + "; post=" + post);
    post = strtem.substr(iter + 1, iter_right - 1 - iter);
    return strtem.substr(0, iter) + strtem.substr(iter_right + 1);
  }
}

std::unordered_map<std::string, std::string> raw_map_split(
    std::string strtem,
    char inner_sp,
    char outer,
    const std::string& default_key) {
  auto re = std::unordered_map<std::string, std::string>();
  auto data = str_split(strtem, outer);
  for (auto& item : data) {
    auto tmp = str_split(item, inner_sp);
    OMNI_ASSERT(tmp.size() == 2 || tmp.size() == 1, "error config: " + strtem);
    if (tmp.size() == 1) {
      if (default_key.empty()) {
        throw std::invalid_argument(
            "error config: " + strtem + ".  default_key is empty for " +
            tmp[0]);
      }
      re[default_key] = tmp[0];
    } else {
      re[tmp[0]] = tmp[1];
    }
  }
  return re;
}

std::unordered_map<std::string, std::string> auto_config_split(
    const std::string& strtem,
    const std::string& default_key) {
  // =
  // const auto num_slash = std::count(strtem.begin(), strtem.end(), '/');
  if (strtem.find('/') != std::string::npos) {
    return map_split(strtem, '=', '/', default_key);
  }
  return map_split(strtem, '=', ',', default_key);
}

namespace {
size_t min3(size_t a, size_t b, size_t c) {
  a = a < b ? a : b;
  return a < c ? a : c;
}
} // namespace
// levenshtein distance
size_t edit_distance(const std::string& s, const std::string& t) {
  size_t dp[s.length() + 1][t.length() + 1];
  for (size_t i = 0; i <= s.length(); i++)
    dp[i][0] = i;
  for (size_t j = 1; j <= t.length(); j++)
    dp[0][j] = j;
  for (size_t j = 0; j < t.length(); j++) {
    for (size_t i = 0; i < s.length(); i++) {
      if (s[i] == t[j])
        dp[i + 1][j + 1] = dp[i][j];
      else
        dp[i + 1][j + 1] =
            min3(dp[i][j + 1] + 1, dp[i + 1][j] + 1, dp[i][j] + 1);
    }
  }
  return dp[s.length()][t.length()];
}

size_t replace_once(
    std::string& str,
    const std::string& from,
    const std::string& to) {
  size_t start_pos = str.find(from);
  if (start_pos != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    return start_pos + to.length(); // 返回替换后的下一个查找位置
  }
  return std::string::npos; // 返回未找到的标志
}

// namespace config_parser {

// Check if all brackets in the string are properly closed
bool areBracketsBalanced(const std::string& str) {
  std::stack<char> stack;
  for (char ch : str) {
    if (ch == '(' || ch == '[' || ch == '{') {
      stack.push(ch);
    } else if (ch == ')' || ch == ']' || ch == '}') {
      if (stack.empty())
        return false;
      char top = stack.top();
      stack.pop();
      if ((ch == ')' && top != '(') || (ch == ']' && top != '[') ||
          (ch == '}' && top != '{')) {
        return false;
      }
    }
  }
  return stack.empty();
}

// Find all valid inner separators (those not inside any brackets)
std::vector<size_t> findValidSeparators(const std::string& str, char sep) {
  std::vector<size_t> validSeparators;
  std::stack<char> bracketStack;

  for (size_t i = 0; i < str.length(); ++i) {
    char ch = str[i];
    if (ch == '(' || ch == '[' || ch == '{') {
      bracketStack.push(ch);
    } else if (ch == ')' || ch == ']' || ch == '}') {
      if (!bracketStack.empty()) {
        bracketStack.pop();
      }
    } else if (ch == sep && bracketStack.empty()) {
      validSeparators.push_back(i);
    }
  }
  return validSeparators;
}

// Function to find valid outer separators between two inner separators
size_t findValidOuterSeparator(
    const std::string& str,
    char outer_sp,
    size_t start,
    size_t end) {
  std::stack<char> bracketStack;
  size_t lastValidPos = std::string::npos;

  for (size_t i = start; i < end; ++i) {
    char ch = str[i];
    if (ch == '(' || ch == '[' || ch == '{') {
      bracketStack.push(ch);
    } else if (ch == ')' || ch == ']' || ch == '}') {
      if (!bracketStack.empty()) {
        bracketStack.pop();
      }
    } else if (ch == outer_sp && bracketStack.empty()) {
      lastValidPos = i;
    }
  }
  return lastValidPos;
}

// Split the string into key-value pairs
std::unordered_map<std::string, std::string> map_split(
    std::string strtem,
    char inner_sp,
    char outer_sp,
    const std::string& default_key,
    bool reverse) {
  remove_space_and_ctrl(strtem);
  if (strtem.empty()) {
    return {};
  }
  std::unordered_map<std::string, std::string> result;

  // Check if brackets are balanced
  if (!areBracketsBalanced(strtem)) {
    throw std::invalid_argument("Unbalanced brackets in the input string.");
  }

  // Find all valid inner separators
  std::vector<size_t> validInnerSeparators =
      findValidSeparators(strtem, inner_sp);

  // If no valid inner separators, treat the whole string as value with
  // default key
  if (validInnerSeparators.empty()) {
    if (default_key.empty()) {
      throw std::invalid_argument(
          "default_key is empty. All key values must be explicitly "
          "provided. IMPUT" +
          strtem);
    }
    result[default_key] = strtem;
    return result;
  }

  std::vector<std::string> key_values;
  // Process the first key-value pair
  OMNI_ASSERT(validInnerSeparators.size() <= 2048 * 2048);
  size_t firstInner = validInnerSeparators[0];
  size_t outerPos = findValidOuterSeparator(strtem, outer_sp, 0, firstInner);
  if (outerPos == std::string::npos) {
    // SPDLOG_INFO(
    //     "No valid outer separator found before the first inner separator. "
    //     "strtem= {}",
    //     strtem);
  }
  key_values.push_back(strtem.substr(0, validInnerSeparators[0]));

  // Process the remaining key-value pairs
  for (size_t i = 1; i < validInnerSeparators.size(); ++i) {
    size_t prevInner = validInnerSeparators[i - 1];
    size_t currInner = validInnerSeparators[i];
    outerPos =
        findValidOuterSeparator(strtem, outer_sp, prevInner + 1, currInner);
    if (outerPos == std::string::npos) {
      throw std::invalid_argument(
          "No valid outer separator found between two inner separators.");
    }
    key_values.push_back(
        strtem.substr(prevInner + 1, outerPos - (prevInner + 1)));
    key_values.push_back(
        strtem.substr(outerPos + 1, currInner - (outerPos + 1)));
  }

  // Process the last key-value pair
  size_t lastInner = validInnerSeparators.back();
  key_values.push_back(strtem.substr(lastInner + 1));

  for (size_t i = 0; i < key_values.size() / 2; ++i) {
    if (reverse)
      result[key_values[i * 2 + 1]] = key_values[i * 2];
    else
      result[key_values[i * 2]] = key_values[i * 2 + 1];
  }

  const std::unordered_set<char> invalid(
      {inner_sp, outer_sp, '{', '}', '[', ']', '(', ')'});
  for (const auto& kv : result) {
    if (kv.first.empty())
      throw std::invalid_argument(
          "Empty key. When providing multiple configuration pairs, each "
          "key must be explicitly provided");
    for (char item : kv.first) {
      if (invalid.count(item) != 0) {
        throw std::invalid_argument("Invalid character in key.");
      }
    }
  }

  return result;
}

// }  // namespace config_parser
} // namespace omniback::str
