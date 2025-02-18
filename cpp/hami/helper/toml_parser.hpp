#pragma once
#include "hami/helper/string.hpp"

namespace hami::toml {
str::mapmap parse_from_file(const std::string& toml_str);
}  // namespace hami::toml
