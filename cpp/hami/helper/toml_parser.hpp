#pragma once
#include "hami/helper/string.hpp"

namespace hami::toml {
str::mapmap parse(const std::string& toml_str);
}  // namespace hami::toml
