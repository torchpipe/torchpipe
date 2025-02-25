// Copyright 2021-2025 NetEase.
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

#ifndef HAMI_COMMON_MACRO_H
#define HAMI_COMMON_MACRO_H

#include "hami/helper/hami_export.h"

namespace {
// constexpr
inline constexpr const char* file_name(const char* path) {
    const char* file = path;
    while (*path) {
        if (*path++ == '/') {
            file = path;
        }
    }
    return file;
}
}  // namespace

#ifdef NEVER_DEFINE_THIS_COMMENTED_CODE
#define HAMI_ASSERT(x, args...)                                          \
    do {                                                                 \
        if (!(x)) {                                                      \
            throw std::runtime_error(                                    \
                "[" + std::string(file_name(__FILE__)) + ":" +           \
                std::to_string(__LINE__) + std::string(" ") +            \
                std::string(__FUNCTION__) + std::string("]: assert `") + \
                std::string(#x) + std::string("` failed. ") +            \
                std::string(args));                                      \
        }                                                                \
    } while (false)
#endif

#define HAMI_ASSERT_V1(x, ...)                                           \
    do {                                                                 \
        if (!(x)) {                                                      \
            throw std::runtime_error(                                    \
                "[" + std::string(file_name(__FILE__)) + ":" +           \
                std::to_string(__LINE__) + std::string(" ") +            \
                std::string(__FUNCTION__) + std::string("]: assert `") + \
                std::string(#x) + std::string("` failed. ") +            \
                std::string(__VA_ARGS__));                               \
        }                                                                \
    } while (false)

#define HAMI_ASSERT(x, ...)                              \
    do {                                                 \
        if (!(x)) {                                      \
            throw std::runtime_error(                    \
                "\nAssertion failed:\n"                  \
                "\tFile: " +                             \
                std::string(file_name(__FILE__)) + ":" + \
                std::to_string(__LINE__) +               \
                "\n"                                     \
                "\tFunction: " +                         \
                std::string(__FUNCTION__) +              \
                "\n"                                     \
                "\tExpression: `" +                      \
                std::string(#x) +                        \
                "`\n"                                    \
                "\tMessage: " +                          \
                std::string(__VA_ARGS__));               \
        }                                                \
    } while (false)

#define HAMI_THROW(args...)                                               \
    {                                                                     \
        throw std::runtime_error("[" + std::string(file_name(__FILE__)) + \
                                 ":" + std::to_string(__LINE__) +         \
                                 std::string(" ") +                       \
                                 std::string(__FUNCTION__) +              \
                                 std::string("]: ") + std::string(args)); \
    }

#define HAMI_FATAL_ASSERT(x, ...) HAMI_ASSERT(x, __VA_ARGS__)
#define IPIPE_ASSERT(x, ...) HAMI_ASSERT(x, __VA_ARGS__)

#define SHUTDOWN_TIMEOUT 500

#define STR_CONFIG_GET(config, key)                                          \
    auto iter = config.find(#key);                                           \
    HAMI_ASSERT(iter != config.end(),                                        \
                "Incomplete configuration: missing " #key " configuration"); \
    const std::string key = iter->second;

#define TRACE_EXCEPT(input)                                                   \
    {                                                                         \
        try {                                                                 \
            input;                                                            \
        } catch (const std::exception& e) {                                   \
            const auto& trace_exception_msg =                                 \
                ('[' + std::string(__FILE__) + ":" +                          \
                 std::to_string(__LINE__) + "]: failed:\n `" +                \
                 std::string(#input) + "`.");                                 \
            throw std::runtime_error(std::string(e.what()) +                  \
                                     "\nerror: " + trace_exception_msg);      \
        } catch (...) {                                                       \
            const auto& trace_exception_msg =                                 \
                ('[' + std::string(__FILE__) + ":" +                          \
                 std::to_string(__LINE__) + "]: failed:\n `" +                \
                 std::string(#input) + "`.");                                 \
            throw std::runtime_error(                                         \
                std::string("an exception not drived from std::exception.") + \
                "\nerror: " + trace_exception_msg);                           \
        }                                                                     \
    }
#endif