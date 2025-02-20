#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <charconv>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <algorithm>

namespace hami::str {

using string = std::string;
using str_map = std::unordered_map<string, string>;
using mapmap = std::unordered_map<string, str_map>;

std::vector<std::string> str_split(std::string strtem, char a);

template <typename T>
std::vector<T> str_split(const std::string& input, char delimiter = ',') {
    static_assert(std::is_arithmetic_v<T>,
                  "T must be an arithmetic type (integer or floating-point)");

    std::vector<T> result;
    std::string trimmed_input = input;

    // Remove all spaces from the input string
    trimmed_input.erase(
        std::remove(trimmed_input.begin(), trimmed_input.end(), ' '),
        trimmed_input.end());

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
                auto [ptr, ec] = std::from_chars(
                    token.data(), token.data() + token.size(), value);
                if (ec == std::errc()) {
                    if (ptr == token.data() + token.size()) {
                        result.push_back(value);
                    } else {
                        throw std::invalid_argument(
                            "Invalid integer at index " +
                            std::to_string(token_index) + ": '" + token +
                            "'. Non-numeric characters found.");
                    }
                } else if (ec == std::errc::result_out_of_range) {
                    throw std::out_of_range("Integer out of range at index " +
                                            std::to_string(token_index) +
                                            ": '" + token + "'");
                } else {
                    throw std::invalid_argument("Invalid integer at index " +
                                                std::to_string(token_index) +
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
std::vector<std::vector<T>> str_split(const std::string& input,
                                      char inner_delimiter,
                                      char outer_delimiter) {
    std::vector<std::vector<T>> result;
    std::vector<std::string> outer_split = str_split(input, outer_delimiter);
    result.reserve(outer_split.size());

    for (const auto& item : outer_split) {
        result.push_back(str_split<T>(item, inner_delimiter));
    }
    return result;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> str_split(std::string input,
                                                   char inner_delimiter,
                                                   char middle_delimiter,
                                                   char outer_delimiter) {
    std::vector<std::vector<std::vector<T>>> result;

    input.erase(std::remove(input.begin(), input.end(), ' '), input.end());
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
std::vector<std::string> items_split(std::string input, char delimiter,
                                     char skipLeft = '[', char skipRight = ']');
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
std::vector<std::string> flatten_brackets(const std::string& strtem,
                                          char left = '[', char right = ']');

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
 * Sequential container supports this functionality.
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
void brackets_split(const std::string& src,
                    std::unordered_map<std::string, std::string>& dst_config,
                    std::string key, char left = '[', char right = ']');

std::string brackets_split(
    const std::string& src,
    std::unordered_map<std::string, std::string>& dst_config, char left = '[',
    char right = ']');

std::string prefix_parentheses_split(const std::string& strtem,
                                     std::string& pre_str);
std::string post_parentheses_split(const std::string& strtem,
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
bool is_comma_separable(const std::string& strtem, char left, char right);

/** @brief
 * split "a=1,2/b=1" to {{"a", "1,2"}, {"b", "1"}} by config_split("a=1,2/b=1",
 * '=', '/') split "a=1,b=2" to {{"a", "1"}, {"b", "2"}} by
 * config_split("a=1,b=2", '=', ',') split "a" to {{"a", "1"}} by
 * config_split("a", '=', ',', "1")
 */
std::unordered_map<std::string, std::string> map_split(
    std::string strtem, char inner_sp, char outer,
    const std::string& default_value);

/**
 * call map_split(strtem, '=', '/', "1") if find '/', else  map_split(strtem,
 * '=',
 * ',', "1")
 */
std::unordered_map<std::string, std::string> auto_config_split(
    const std::string& strtem);

size_t edit_distance(const std::string& s, const std::string& t);

template <typename T>
void str2int(const str::str_map& config, const std::string& key,
             T& default_value) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    auto iter = config.find(key);
    if (iter == config.end()) {
        SPDLOG_DEBUG("parameter {} not found, default to {}", key,
                     default_value);
        return;
    }

    const std::string& value = iter->second;
    auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(),
                                     default_value);
    bool success = ec == std::errc() && ptr == value.data() + value.size();
    if (!success) {
        throw std::invalid_argument("invalid " + key + ": " + value);
    }
}

template <typename T>
T str2int(const str::str_map& config, const std::string& key) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    auto iter = config.find(key);
    if (iter == config.end()) {
        throw std::invalid_argument("parameter " + key + " not found");
    }

    T result_value;
    const std::string& value = iter->second;
    auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(),
                                     result_value);
    bool success = ec == std::errc() && ptr == value.data() + value.size();
    if (!success) {
        throw std::invalid_argument("invalid " + key + ": " + value);
    }
    return result_value;
}

template <typename T>
T update(const str::str_map& config, const std::string& key) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    auto iter = config.find(key);
    if (iter == config.end()) {
        throw std::invalid_argument("parameter " + key + " not found");
    }

    T result_value;
    const std::string& value = iter->second;
    auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(),
                                     result_value);
    bool success = ec == std::errc() && ptr == value.data() + value.size();
    if (!success) {
        throw std::invalid_argument("invalid " + key + ": " + value);
    }
    return result_value;
}

static inline std::string get_str(const str::str_map& config,
                                  const std::string& key) {
    auto iter = config.find(key);
    if (iter == config.end()) {
        throw std::invalid_argument("parameter " + key + " not found");
    }

    return iter->second;
}

static inline std::string join(const std::unordered_set<std::string>& vec,
                               char sep) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& s : vec) {
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

static inline void try_update(
    const str::str_map& config, const std::string& key,
    std::string& default_value,
    const std::unordered_set<std::string>& valid_inputs = {}) {
    auto iter = config.find(key);
    if (iter == config.end()) {
        return;
    }

    if (valid_inputs.count(iter->second) == 0) {
        throw std::invalid_argument(
            "parameter " + key + " is out of range: " + iter->second +
            ", valid inputs are " + str::join(valid_inputs, ','));
    }

    default_value = iter->second;
}

template <typename T>
void try_update(const str::str_map& config, const std::string& key,
                T& default_value) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    auto iter = config.find(key);
    if (iter == config.end()) {
        SPDLOG_DEBUG("parameter {} not found, default to {}", key,
                     default_value);
        return;
    }

    const std::string& value = iter->second;
    auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(),
                                     default_value);
    bool success = ec == std::errc() && ptr == value.data() + value.size();
    if (!success) {
        throw std::invalid_argument("invalid " + key + ": " + value);
    }
}

}  // namespace hami::str
namespace hami {

namespace str {

template <typename... Args>
std::string format(const std::string& fmt, Args&&... args) {
    std::ostringstream oss;
    std::string remaining_fmt = fmt;  // 引入一个非 const 的局部变量
    size_t index = 0;
    size_t pos = 0;

    // 使用 lambda 捕获参数包
    auto format_arg = [&](const auto& arg) {
        // 如果索引超过参数的数量，直接返回
        if (index >= sizeof...(Args)) {
            return;
        }

        std::string placeholder = "{" + std::to_string(index) + "}";
        size_t current_pos = remaining_fmt.find(placeholder, pos);
        if (current_pos != std::string::npos) {
            // 将当前占位符之前的部分添加到 ostringstream
            oss << remaining_fmt.substr(0, current_pos);
            // 添加参数
            oss << arg;
            // 更新剩余格式字符串
            remaining_fmt =
                remaining_fmt.substr(current_pos + placeholder.length());
            pos = 0;  // 重置位置，以确保后续搜索从头开始
        } else {
            // 如果没有找到占位符，直接添加剩余部分
            oss << remaining_fmt;
            remaining_fmt.clear();
        }
        index++;
    };

    // 依次处理每个参数
    (format_arg(std::forward<Args>(args)), ...);

    // 添加剩余部分
    oss << remaining_fmt;
    return oss.str();
}

bool replace_once(std::string& str, const std::string& from,
                  const std::string& to);

}  // namespace str

}  // namespace hami
