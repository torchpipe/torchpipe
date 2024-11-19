// Copyright 2021-2024 NetEase.
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

#pragma once

#include <string>
#include <set>
#include <unordered_map>
#include "Backend.hpp"
#include "dict.hpp"
namespace ipipe {
// std::string toml2str(toml::value v);

/// 解析toml文件到双层map中。
mapmap parse_toml(std::string toml_path);
void handle_config(mapmap& config);
void update_global(mapmap& config);
std::set<std::string> handle_ring(const mapmap& config_param);
void register_config(std::string config, std::string value);
int get_registered_config(std::string config, std::string item, int default_value);

}  // namespace ipipe
