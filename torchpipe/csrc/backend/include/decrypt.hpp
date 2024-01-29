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
// 参考github代码：
// https://gist.github.com/irbull/c76a8c60e049a9fcba1116aa81771253

// default command
// openssl enc -aes-256-cbc -in resnet18.py -out resnet18.py.encrypt -pass
// pass:your_password   -md sha1

// if you use -pbkdf2 in your command as follow  ,you should set pbkdf2 = 1 in
// aes_256_decrypt fuction!!
// openssl enc -aes-256-cbc -in resnet18.py -out resnet18.py.encrypt -pass
// pass:your_password  -md sha1  -pbkdf2 -iter 1 -p

#include <vector>
#include <fstream>
#include "string.h"

#ifdef USE_DECRYPT

#include "openssl/aes.h"
#include "openssl/modes.h"
#include "openssl/conf.h"
#include "openssl/evp.h"
#include "openssl/err.h"

namespace ipipe {

constexpr auto ENCRYPT_KEY = "grasslands";

void handleOpenSSLErrors(void);

void initAES(const std::string& pass, unsigned char* salt, unsigned char* key, unsigned char* iv,
             int pbkdf2);

bool aes_256_decrypt_from_cache(unsigned char* input, size_t length,
                                std::vector<char>& decrypt_data, int pbkdf2 = 0,
                                const std::string& password = ENCRYPT_KEY);

bool aes_256_decrypt_from_path(const std::string& input, std::vector<char>& decrypt_data,
                               int pbkdf2 = 0, const std::string& password = ENCRYPT_KEY);

};  // namespace ipipe
#else

bool aes_256_decrypt_from_cache(unsigned char* input, size_t length,
                                std::vector<char>& decrypt_data, int pbkdf2 = 0,
                                const std::string& password = "") {
  throw std::runtime_error("not implemented.");
}
bool aes_256_decrypt_from_path(const std::string& input, std::vector<char>& decrypt_data,
                               int pbkdf2 = 0, const std::string& password = "") {
  throw std::runtime_error("not implemented.");
}
#endif