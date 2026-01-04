#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace omniback::str {
const auto ITEM_DELIMITERS = std::unordered_set<char>{
    ',',
    ';',
    '/',
};

using string = std::string;
using str_map = std::unordered_map<string, string>;
using mapmap = std::unordered_map<string, str_map>;

std::vector<std::string> str_split(std::string strtem, char a);
void remove_space_and_ctrl(std::string& strtem);

template <typename T>
std::vector<T> str_split(const std::string& input, char delimiter = ',') {
  static_assert(
      std::is_arithmetic_v<T>,
      "T must be an arithmetic type (integer or floating-point)");

  std::vector<T> result;
  std::string trimmed_input = input;

  remove_space_and_ctrl(trimmed_input);

  if (trimmed_input.empty()) {
    return result;
  }

  std::istringstream iss(trimmed_input);
  std::string token;
  int token_index = 0;

  while (std::getline(iss, token, delimiter)) {
    if (!token.empty()) {
      T value;

      if constexpr (std::is_integral_v<T>) {
        auto [ptr, ec] =
            std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec == std::errc()) {
          if (ptr == token.data() + token.size()) {
            result.push_back(value);
          } else {
            throw std::invalid_argument(
                "Invalid integer at index " + std::to_string(token_index) +
                ": '" + token + "'. Non-numeric characters found.");
          }
        } else if (ec == std::errc::result_out_of_range) {
          throw std::out_of_range(
              "Integer out of range at index " + std::to_string(token_index) +
              ": '" + token + "'");
        } else {
          throw std::invalid_argument(
              "Invalid integer at index " + std::to_string(token_index) +
              ": '" + token + "'");
        }
      } else if constexpr (std::is_floating_point_v<T>) {
        char* end;
        value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size()) {
          throw std::invalid_argument(
              "Invalid floating-point value at index " +
              std::to_string(token_index) + ": '" + token +
              "'. Non-numeric characters found.");
        }
        if (errno == ERANGE) {
          throw std::out_of_range(
              "Floating-point value out of range at index " +
              std::to_string(token_index) + ": '" + token + "'");
        }
        result.push_back(value);
      }
    }
    token_index++;
  }

  return result;
}

template <typename T>
std::vector<std::vector<T>> str_split(
    const std::string& in_input,
    char inner_delimiter,
    char outer_delimiter) {
  auto input = in_input;
  remove_space_and_ctrl(input);
  std::vector<std::vector<T>> result;
  std::vector<std::string> outer_split = str_split(input, outer_delimiter);
  result.reserve(outer_split.size());

  for (const auto& item : outer_split) {
    result.push_back(str_split<T>(item, inner_delimiter));
  }
  return result;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> str_split(
    std::string input,
    char inner_delimiter,
    char middle_delimiter,
    char outer_delimiter) {
  std::vector<std::vector<std::vector<T>>> result;

  remove_space_and_ctrl(input);

  if (input.empty()) {
    return result;
  }

  std::vector<std::string> outer_split = str_split(input, outer_delimiter);
  result.reserve(outer_split.size());

  for (const auto& outer_item : outer_split) {
    std::vector<std::vector<T>> middle_result =
        str_split<T>(outer_item, inner_delimiter, middle_delimiter);

    result.push_back(std::move(middle_result));
  }

  return result;
}

/**
 * @brief Splits a string by a delimiter while skipping nested sections.
 *
 * This function splits a string by a specified delimiter, but skips sections
 * enclosed by specified left and right characters. For example,
 * "A,B[C,D(a=1,b=2)],E" will be split into {"A", "B[C,D(a=1,b=2)]", "E"}.
 *
 * @param input The input string to be split.
 * @param delimiter The character used to split the string.
 * @param skipLeft The character that marks the beginning of a section to skip.
 * @param skipRight The character that marks the end of a section to skip.
 * @return A vector of strings after splitting.
 */
std::vector<std::string> items_split(
    std::string input,
    char delimiter,
    char skipLeft = '[',
    char skipRight = ']');
/**
 * @brief Expands nested brackets in strings.
 *
 * This function processes a string containing nested square brackets and
 * expands them according to the following rules:
 * 1. Converts "A[B[C]]" to "A; B; C".
 * 2. Converts "A[B[C],D,B[E[Z1,Z2]]]" to "A; B[C],D,B[E[Z1,Z2]]".
 *
 * Note that in the second case, "B[C],D,B[E[Z1,Z2]]" is treated as a single
 * entity.
 *
 * @note All spaces will be removed from the input string.
 *
 * @param strtem The input string containing brackets.
 * @param left The left bracket character, default is '['.
 * @param right The right bracket character, default is ']'.
 * @return A vector of strings after expanding the brackets.
 */
std::vector<std::string> flatten_brackets(
    const std::string& strtem,
    char left = '[',
    char right = ']');

/**
 * @brief Converts square bracket expressions into key-value pair parameters.
 * For example:
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.py}
 * # From
 * {"backend":"A[B[C]]"}
 * # Expands to
 * {"backend":"A","A::dependency":"B","B::dependency":"C"}
 * # From
 * {"backend":"A[B[C],D,B[E[Z1,Z2]]"}
 * # Expands to
 * {"backend":"A","A::dependency":"B[C],D,B[E[Z1,Z2]]"}
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Here, the comma operator is parsed as a whole, `B[C],D,B[E[Z1,Z2]]` is
 * treated as a valid backend of A. The validity depends on A being able to
 * understand the three backends `B[C]`, `D`, and `B[E[Z1,Z2]]` by calling the
 * brackets_split function to expand them into {backend=B, B::dependency=C},
 * {backend=D}, and {backend=B, B::dependency=E[Z1,Z2]}. The default @ref
 * SequentialV0 container supports this functionality.
 *
 * Refer to @ref flatten_brackets.
 * @warning Does not support cases that cause duplicate keys. For example,
 * B[B[C],D,B[E[Z1,Z2]]] and B[C[B]] are supported,
 * but B[B[C]] is not supported because the latter will have duplicate
 * B::dependency keys, and an invalid_argument exception will be thrown.
 * @param src The input string containing square bracket expressions.
 * @param dst_config The output unordered_map to store the key-value pairs.
 * @param key The initial key for the backend, default is "backend".
 * @param left The left bracket character, default is '['.
 * @param right The right bracket character, default is ']'.
 */
void brackets_split(
    const std::string& src,
    std::unordered_map<std::string, std::string>& dst_config,
    std::string key,
    char left = '[',
    char right = ']');

std::string brackets_split(
    const std::string& src,
    std::unordered_map<std::string, std::string>& dst_config,
    char left = '[',
    char right = ']');

std::string prefix_parentheses_split(
    const std::string& strtem,
    std::string& pre_str);
std::string post_parentheses_split(
    const std::string& strtem,
    std::string& post_str);

/**
 * @brief Checks if the left and right brackets in the string are matched and if
 * the string can be separated by outer commas.
 *
 * @param strtem The input string to check.
 * @param left The left bracket character.
 * @param right The right bracket character.
 * @return true If the string can be separated by commas, e.g.,
 * B[C],D,B[E[Z1,Z2]] can be separated into three parts.
 * @return false If the string cannot be separated by commas, e.g.,
 * A[B[C],D,B[E[Z1,Z2]]] cannot be separated by commas.
 * @exception Throws invalid_argument exception if brackets are unmatched or if
 * a comma appears at the first or last position of the string.
 */
bool is_comma_semicolon_separable(
    const std::string& strtem,
    char left,
    char right);

/** @brief
 * split "a=1,2/b=1" to {{"a", "1,2"}, {"b", "1"}} by config_split("a=1,2/b=1",
 * '=', '/') split "a=1,b=2" to {{"a", "1"}, {"b", "2"}} by
 * config_split("a=1,b=2", '=', ',') split "a" to {{"a", "1"}} by
 * config_split("a", '=', ',', "1")
 *
 *
 * keya=A[B,C(a=d)](a,d){a=d},D,keyb=E => keya="A[B,C(a=d)](a,d){a=d},D" and
 * keyb=E
 */
// std::unordered_map<std::string, std::string> map_split(
//     std::string strtem, char inner_sp, char outer,
//     const std::string& default_value);

/**
 * call map_split(strtem, '=', '/', "1") if find '/', else  map_split(strtem,
 * '=',
 * ',', "1")
 */
std::unordered_map<std::string, std::string> auto_config_split(
    const std::string& strtem,
    const std::string& default_key = "");

size_t edit_distance(const std::string& s, const std::string& t);

// template <typename T>
// void str2int(const str::str_map &config, const std::string &key,
//              T &default_value) {
//     static_assert(std::is_integral_v<T>, "T must be an integral type");
//     auto iter = config.find(key);
//     if (iter == config.end()) {
//         SPDLOG_DEBUG("parameter {} not found, default to {}", key,
//                      default_value);
//         return;
//     }

//     const std::string &value = iter->second;
//     auto [ptr, ec] = std::from_chars(value.data(), value.data() +
//     value.size(),
//                                      default_value);
//     bool success = ec == std::errc() && ptr == value.data() + value.size();
//     if (!success) {
//         throw std::invalid_argument("invalid " + key + ": " + value);
//     }
// }

template <typename T>
T str2int(const str::str_map& config, const std::string& key) {
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  auto iter = config.find(key);
  if (iter == config.end()) {
    throw std::invalid_argument("parameter " + key + " not found");
  }

  T result_value;
  const std::string& value = iter->second;
  auto [ptr, ec] =
      std::from_chars(value.data(), value.data() + value.size(), result_value);
  bool success = ec == std::errc() && ptr == value.data() + value.size();
  if (!success) {
    throw std::invalid_argument("invalid " + key + ": " + value);
  }
  return result_value;
}

template <typename>
struct always_false : std::false_type {};

template <typename T = std::string>
T get(const str::str_map& config, const std::string& key) {
  auto iter = config.find(key);
  if (iter == config.end()) {
    throw std::invalid_argument("Parameter '" + key + "' not found");
  }

  const std::string& value = iter->second;

  if constexpr (std::is_integral_v<T>) {
    T result;
    auto [ptr, ec] =
        std::from_chars(value.data(), value.data() + value.size(), result);

    if (ec != std::errc() || ptr != value.data() + value.size()) {
      throw std::invalid_argument(
          "Invalid integer value for '" + key + "': " + value);
    }
    return result;
  } else if constexpr (std::is_same_v<T, std::string>) {
    return value; // 直接返回字符串
  } else {
    static_assert(
        always_false<T>::value,
        "Unsupported type: must be integral or std::string");
  }
}

static inline std::string get_str(
    const str::str_map& config,
    const std::string& key) {
  auto iter = config.find(key);
  if (iter == config.end()) {
    throw std::invalid_argument("parameter " + key + " not found");
  }

  return iter->second;
}

template <typename T1, typename T>
std::string join(
    const std::vector<std::pair<T1, T>>& container,
    char sep = ',') {
  std::ostringstream oss;
  bool first = true;
  for (const auto& s : container) {
    if (!first) {
      oss << sep;
    }
    oss << s.first;
    first = false;
  }
  return oss.str();
}

template <typename Container>
static inline std::string join(const Container& container, char sep = ',') {
  static_assert(
      std::is_same_v<typename Container::value_type, std::string>,
      "Container must hold std::string elements");

  std::ostringstream oss;
  bool first = true;
  for (const auto& s : container) {
    if (!first) {
      oss << sep;
    }
    oss << s;
    first = false;
  }
  return oss.str();
}

static inline std::string tolower(std::string s) {
  constexpr auto DIFF_UPPER_LOWER = 'a' - 'A';
  for (char& ch : s) {
    if (ch >= 'A' && ch <= 'Z') {
      ch += DIFF_UPPER_LOWER;
    }
  }
  return s;
}

template <typename T>
void try_update(
    const str::str_map& config,
    const std::string& key,
    T& default_value,
    const std::unordered_set<std::string>& valid_inputs = {}) {
  auto iter = config.find(key);
  if (iter == config.end()) {
    return;
  }

  const std::string& value = iter->second;

  if constexpr (std::is_same_v<T, std::string>) {
    // 字符串类型校验逻辑
    if (!valid_inputs.empty() && !valid_inputs.count(value)) {
      throw std::invalid_argument(
          "Parameter " + key + " value `" + value +
          "` is invalid. "
          "Valid options: [" +
          str::join(valid_inputs, ',') + "]");
    }
    default_value = value;

  } else {
    // 数值类型转换逻辑
    static_assert(
        std::is_arithmetic_v<T>, "T must be string or arithmetic type");

    T parsed_value;
    auto [ptr, ec] = std::from_chars(
        value.data(), value.data() + value.size(), parsed_value);

    if (ec != std::errc() || ptr != value.data() + value.size()) {
      throw std::invalid_argument(
          "Failed to parse " + key + " as " + typeid(T).name() +
          ". value=" + value);
    }
    default_value = parsed_value;

    // 显式忽略未使用的 valid_inputs 避免警告
    (void)valid_inputs;
  }
}

} // namespace omniback::str
namespace omniback {

namespace str {
template <typename... Args>
std::string format(const std::string& fmt, Args&&... args) {
  std::ostringstream oss;
  std::string remaining_fmt = fmt;
  size_t current_index = 0; // 当前要替换的参数索引

  // 遍历所有参数
  auto replace_placeholder = [&](const auto& arg) {
    // 构造当前占位符，如 "{0}", "{1}", ...
    std::string placeholder = "{" + std::to_string(current_index) + "}";
    size_t pos = remaining_fmt.find(placeholder);

    if (pos != std::string::npos) {
      // 找到占位符，替换它
      oss << remaining_fmt.substr(0, pos); // 写入占位符之前的部分
      oss << arg; // 写入参数
      remaining_fmt =
          remaining_fmt.substr(pos + placeholder.length()); // 更新剩余字符串
    } else {
      // 没有找到占位符，直接写入参数（可能多余，取决于需求）
      // 这里选择忽略多余的参数（不写入）
      // 如果需要严格检查，可以抛出异常
    }

    current_index++; // 移动到下一个参数
  };

  // 先替换所有占位符对应的参数
  (replace_placeholder(std::forward<Args>(args)), ...);

  // 如果还有剩余未替换的 `{N}` 占位符（如 `{3}` 但只有 2 个参数），可以选择：
  // 1. 忽略（当前实现）
  // 2. 抛出异常（更严格）
  // 3. 保留原占位符（不推荐）

  // 最后写入剩余未解析的字符串（如普通文本）
  oss << remaining_fmt;

  return oss.str();
}

size_t replace_once(
    std::string& str,
    const std::string& from,
    const std::string& to);

template <typename T>
std::string vec2str(const std::vector<T>& vec) {
  static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i < vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

static inline bool endswith(const std::string& str, const std::string& suffix) {
  if (suffix.size() > str.size()) {
    return false;
  }
  return std::equal(str.end() - suffix.size(), str.end(), suffix.cbegin());
}
// namespace config_parser {

// Find all valid  separators (those not inside any brackets)
std::vector<size_t> findValidSeparators(const std::string& str, char sep);

/**
 * @brief Split a string into key-value pairs
 *
 * Parses strings like "a=b,c=d" into a map {a:b, c:d}
 * - inner_sp is the character that separates keys from values (e.g., '=')
 * - outer_sp is the character that separates key-value pairs (e.g., ',')
 * - Separators inside brackets are not treated as separators; entire
 * bracketed content is treated as a single unit
 * - Special cases like "a=b,c,c=d" result in {a:"b,c", c:d}
 * - If no valid inner_sp is found outside brackets, returns
 *   {default_key:strtem} if !default_key.empty(), else throws exception
 *
 * @param strtem Input string to parse
 * @param inner_sp Character separating keys from values (e.g. '=')
 * @param outer_sp Character separating key-value pairs (e.g. ',')
 * @param default_key Default key to use when inner separator is missing
 * @return std::unordered_map<std::string, std::string> Resulting key-value
 * map
 * @throws std::invalid_argument if no inner separator is found and
 * default_key is empty
 */
std::unordered_map<std::string, std::string> map_split(
    std::string strtem,
    char inner_sp,
    char outer_sp,
    const std::string& default_key = "",
    bool reverse = false);
// }  // namespace config_parser
} // namespace str

} // namespace omniback
