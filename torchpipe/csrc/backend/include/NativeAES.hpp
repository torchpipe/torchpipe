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

#include "ipipe_common.hpp"
namespace ipipe {

IPIPE_LOCAL std::string decrypt_data(std::string& model_type, std::string data);
IPIPE_LOCAL void encrypt_file_to_file(std::string file_path, std::string out_file_path,
                                      std::string key);
IPIPE_LOCAL void encrypt_buffer_to_file(const std::string& data, std::string out_file_path,
                                        std::string key);
}  // namespace ipipe