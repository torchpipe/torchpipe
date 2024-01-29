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


#include "file_utils.hpp"

#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <fstream>

#include "fs.hpp"
namespace ipipe {
std::vector<std::string> os_listdir(const char* path, bool only_regular_file) {
  std::vector<std::string> results;
  for (const auto& entry : fs::directory_iterator(path)) {
    auto p = entry.path();
    if (only_regular_file && !fs::is_regular_file(entry)) {
      continue;
    }
    results.push_back(p.string());
  }
  return results;
}

bool os_path_exists(const std::string& file_path) {
  std::ifstream file(file_path.c_str());
  return file.good();
}
}  // namespace ipipe