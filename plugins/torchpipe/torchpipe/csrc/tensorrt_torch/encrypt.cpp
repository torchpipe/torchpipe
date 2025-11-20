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


#include <fstream>

#include <memory>
#include <cassert>
#include <vector>

#include "encrypt.hpp"
#include "aes.h"


namespace {


#define TO_STR_INNER(x) #x
#define TO_STR(x) TO_STR_INNER(x)

OMNI_LOCAL std::string get_torchpipe_key() {
  std::string result = TO_STR(SECRET_KEY);

  while (result.size() < 16) {
    result += result;
  }
  return result;
}



template <typename T, std::size_t N>
constexpr uint32_t array_sum(T (&array)[N]) {
  uint32_t sum = 0;
  for (std::size_t i = 0; i < N; i++) {
    sum += array[i];
  }
  return sum;
};

const uint32_t array_sum(std::string in) {
  uint32_t sum = 0;
  for (std::size_t i = 0; i < in.size(); i++) {
    sum += in[i];
  }
  return sum;
};



class OMNI_LOCAL EncryptHelper {
 public:
  std::string encrypt(const char* buffer,size_t buffer_len,  std::string key) {
    if (key.empty()) {
      key = get_torchpipe_key();
    }

    PaddingHead header;
    header.tag = get_torchpipe_tag();
    int total_len =
        4 * 16 - buffer_len % 16 + buffer_len + (sizeof(PaddingHead) % 16) * 16;
    header.data_start = total_len - buffer_len;

    header.data_len = buffer_len;

    std::vector<unsigned char> tmp_vector(total_len);
    assert(tmp_vector.size() % 16 == 0);
    std::memcpy(tmp_vector.data(), &header, sizeof(PaddingHead));
    std::memcpy(tmp_vector.data() + header.data_start, buffer, buffer_len);

    AES aes;
    auto re = aes.EncryptECB(
        tmp_vector, std::vector<unsigned char>(key.begin(), key.end()));

    return std::string(re.begin(), re.end());
  }

  std::vector<unsigned char> decrypt(const std::string& buffer, std::string key) {
    if (key.empty()) {
      key = get_torchpipe_key();
    }

    AES aes;
    auto result = aes.DecryptECB(
        std::vector<unsigned char>(buffer.begin(), buffer.end()),
        std::vector<unsigned char>(key.begin(), key.end()));
    assert(result.size() % 16 == 0);

    return get_data(result);
  }

 private:
  struct PaddingHead {
    char not_used[8];
    uint32_t data_start;
    uint32_t data_len;
    uint32_t tag;
    char time[32] = "2025-11-20";
    char not_used_post[8];
  };
  std::vector<unsigned char> get_data(const std::vector<unsigned char>& data) {
    std::vector<unsigned char> result;
    if (data.size() < sizeof(PaddingHead)) {
      throw std::runtime_error("data.size() < sizeof(head) ");
    }
    const PaddingHead* header =
        reinterpret_cast<const PaddingHead*>(data.data());
    if (header->tag != get_torchpipe_tag() ||
        header->data_start + header->data_len != data.size()) {
      throw std::runtime_error("DECRYPT: tag or version not match.");
    }
    auto* p_start = data.data() + header->data_start;
    result = std::vector<unsigned char>(p_start, p_start + header->data_len);
    return result;
  }
  
  uint32_t get_torchpipe_tag() {
    return 5432023 + sizeof(PaddingHead)*array_sum(get_torchpipe_key()) ;
  }
};
} // namespace

namespace torchpipe {
 
OMNI_LOCAL std::vector<unsigned char> decrypt_file(std::string path) {
  std::ifstream ff(path);
  if (!ff.good()) {
    throw std::runtime_error("open failed: " + path);
  }
  std::string buffer(
      (std::istreambuf_iterator<char>(ff)), std::istreambuf_iterator<char>());

  EncryptHelper decry;
  std::vector<unsigned char> result = decry.decrypt(buffer, "");

  return result;
}

OMNI_LOCAL void encrypt2file(
    const char* data,
    size_t data_len,
    std::string out_file_path) {


  EncryptHelper decry;
  auto re = decry.encrypt(data, data_len, "");
  std::ofstream out_ff(out_file_path);
  out_ff << re;
  return;

}

} // namespace torchpipe
