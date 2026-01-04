#pragma once
#include "omniback/helper/string.hpp"

namespace omniback::toml {
str::mapmap parse(const std::string& toml_str);
} // namespace omniback::toml
