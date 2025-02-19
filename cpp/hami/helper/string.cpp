#include <algorithm>
#include <string>
#include "hami/helper/string.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"

namespace hami::str {

std::vector<std::string> str_split(std::string strtem, char a) {
    std::vector<std::string> strvec;
    if (strtem.empty()) return {};

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

std::vector<std::string> items_split(std::string strtem, char a, char left,
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

std::vector<std::string> flatten_brackets(const std::string& strtem_, char left,
                                          char right) {
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
            HAMI_ASSERT(iter_end == strtem.size() - 1,
                        "brackets not match: " + strtem_);

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

void brackets_split(const std::string& strtem_,
                    std::unordered_map<std::string, std::string>& config,
                    std::string key, char left, char right) {
    auto re = brackets_split(strtem_, config, left, right);
    config[key] = re;
}

std::string brackets_split(const std::string& strtem_,
                           std::unordered_map<std::string, std::string>& config,
                           char left, char right) {
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
            SPDLOG_ERROR("Recursive backend({}) is not allowed. backend={}",
                         brackets[i - 1], strtem_);

            throw std::invalid_argument("Recursive backend(" + brackets[i - 1] +
                                        ") is not allowed. Backend is " +
                                        strtem_);
        }
        new_config[brackets[i - 1] + "::dependency"] = brackets[i];
    }
    for (auto iter = new_config.begin(); iter != new_config.end(); ++iter) {
        config[iter->first] = iter->second;
    }
    return brackets[0];
}

std::string prefix_parentheses_split(const std::string& strtem,
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

std::string post_parentheses_split(const std::string& strtem,
                                   std::string& post) {
    constexpr auto left = '(';
    constexpr auto right = ')';
    auto iter = strtem.find(left);
    auto iter_right = strtem.find_last_of(right, strtem.find('['));

    if (iter == std::string::npos) {
        HAMI_ASSERT(iter_right == std::string::npos);
        return strtem;
    } else {
        HAMI_ASSERT(iter != 0);
        HAMI_ASSERT(iter_right != std::string::npos);
        post = strtem.substr(iter + 1, iter_right - 1 - iter);
        return strtem.substr(0, iter) + strtem.substr(iter_right + 1);
    }
}

std::unordered_map<std::string, std::string> map_split(
    std::string strtem, char inner_sp, char outer,
    const std::string& default_value) {
    auto re = std::unordered_map<std::string, std::string>();
    auto data = str_split(strtem, outer);
    for (auto& item : data) {
        auto tmp = str_split(item, inner_sp);
        HAMI_ASSERT(tmp.size() == 2 || tmp.size() == 1,
                    "error config: " + strtem);
        if (tmp.size() == 1) {
            re[tmp[0]] = default_value;
        } else {
            re[tmp[0]] = tmp[1];
        }
    }
    return re;
}

std::unordered_map<std::string, std::string> auto_config_split(
    const std::string& strtem) {
    // =
    // const auto num_slash = std::count(strtem.begin(), strtem.end(), '/');
    if (strtem.find('/') != std::string::npos) {
        return map_split(strtem, '=', '/', "1");
    }
    return map_split(strtem, '=', ',', "1");
}

namespace {
size_t min3(size_t a, size_t b, size_t c) {
    a = a < b ? a : b;
    return a < c ? a : c;
}
}  // namespace
// levenshtein distance
size_t edit_distance(const std::string& s, const std::string& t) {
    size_t dp[s.length() + 1][t.length() + 1];
    for (size_t i = 0; i <= s.length(); i++) dp[i][0] = i;
    for (size_t j = 1; j <= t.length(); j++) dp[0][j] = j;
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

}  // namespace hami::str

