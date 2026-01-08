#pragma once
#include "omniback/helper/string.hpp"

namespace om::toml {
str::mapmap parse(const std::string& toml_str);
} // namespace om::toml
