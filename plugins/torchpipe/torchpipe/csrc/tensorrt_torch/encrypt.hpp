// Copyright 2021-2023 NetEase.
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

#define OMNI_LOCAL __attribute__((visibility("hidden")))

namespace torchpipe {
 
OMNI_LOCAL std::vector<unsigned char> decrypt_file(std::string path);
// OMNI_LOCAL void encrypt_file_to_file(
//     std::string file_path,
//     std::string out_file_path,
//     std::string key);
OMNI_LOCAL void encrypt2file(
    const char* data,
    size_t data_len,
    std::string out_file_path);

} // namespace torchpipe
