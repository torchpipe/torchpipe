#pragma once
#include <string>
#include <unordered_set>

namespace torchpipe {
constexpr auto TASK_COLOR_KEY = "color";
static const std::unordered_set<std::string> VALID_COLOR_SPACE = {"rgb", "bgr"};

}