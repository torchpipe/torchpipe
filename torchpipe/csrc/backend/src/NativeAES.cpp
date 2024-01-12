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

#include "NativeAES.hpp"
#include "AES.h"
#include "base_logging.hpp"
#include <fstream>
#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"

#include <memory>

namespace {

#ifdef IPIPE_KEY
#define TO_STR_INNER(x) #x
#define TO_STR(x) TO_STR_INNER(x)
IPIPE_LOCAL std::string get_ipipe_key() {
  std::string result = TO_STR(IPIPE_KEY);
  IPIPE_ASSERT(result.size() > 8);
  if (result.size() > 16) {
    result.resize(16);
  }
  while (result.size() < 16) {
    result += result.back();
  }
  return result;
}
#else

IPIPE_LOCAL std::string get_ipipe_key() {
  throw ::std::runtime_error(
      "If you need to use Encrypt/decrypt capability, you must set the environment "
      "variable IPIPE_KEY or preprocessor macro  `-DIPIPE_KEY` when compiling TorchPipe. for "
      "example, `IPIPE_KEY=j987hyuihk "
      "python setup.py bdist_wheel`.");
  return "";
}
#endif

constexpr auto ipipe_version = "0.x.x";

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

constexpr int ipipe_version_sum() {
  int sum = 0;
  for (std::size_t i = 0; i < sizeof(ipipe_version); i++) {
    sum += ipipe_version[i];
  }
  return sum;
};

class IPIPE_LOCAL EncryptHelper {
 public:
  std::string encrypt(const std::string& buffer, std::string key) {
    if (key.empty()) {
      key = get_ipipe_key();
    }
    if (key.size() < 16) {
      SPDLOG_ERROR("length of key should >= 16");
      throw std::runtime_error("length of key should >= 16");
    }
    PaddingHead header;
    header.tag = get_ipipe_tag();
    int total_len = 4 * 16 - buffer.size() % 16 + buffer.size() + (sizeof(PaddingHead) % 16) * 16;
    header.data_start = total_len - buffer.size();

    header.data_len = buffer.size();

    std::vector<unsigned char> tmp_vector(total_len);
    assert(tmp_vector.size() % 16 == 0);
    std::memcpy(tmp_vector.data(), &header, sizeof(PaddingHead));
    std::memcpy(tmp_vector.data() + header.data_start, buffer.data(), buffer.size());

    AES aes;
    auto re = aes.EncryptECB(tmp_vector, std::vector<unsigned char>(key.begin(), key.end()));

    return std::string(re.begin(), re.end());
  }

  std::string decrypt(const std::string& buffer, std::string key) {
    if (key.empty()) {
      key = get_ipipe_key();
    }
    if (key.size() < 16) {
      SPDLOG_ERROR("length of key should >= 16");
      throw std::runtime_error("length of key should >= 16");
    }
    AES aes;
    auto result = aes.DecryptECB(std::vector<unsigned char>(buffer.begin(), buffer.end()),
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
    char version[sizeof("3ewrfd")] = "3ewrfd";
    char not_used_post[8];
  };
  std::string get_data(const std::vector<unsigned char>& data) {
    std::string result;
    if (data.size() < sizeof(PaddingHead)) {
      throw std::runtime_error("data.size() < sizeof(head) ");
    }
    const PaddingHead* header = reinterpret_cast<const PaddingHead*>(data.data());
    if (header->tag != get_ipipe_tag() || header->data_start + header->data_len != data.size()) {
      SPDLOG_ERROR("decrypt: tag or version not match.");
      throw std::runtime_error("tag or version not match.");
    }
    auto* p_start = data.data() + header->data_start;
    result = std::string(p_start, p_start + header->data_len);
    return result;
  }
  uint32_t get_ipipe_tag() {
    return 54320000 + sizeof(PaddingHead) + array_sum(get_ipipe_key());  // + ipipe_version_sum();
  }
};
}  // namespace

namespace ipipe {

// #ifdef IPIPE_KEY
IPIPE_LOCAL std::string decrypt_data(std::string& model_type, std::string data) {
  if (endswith(model_type, ".encrypted")) {
    std::ifstream ff(data);
    if (!ff.good()) {
      SPDLOG_ERROR("open {} failed ", data);
      throw std::runtime_error("open failed: " + data);
    }
    std::string buffer((std::istreambuf_iterator<char>(ff)), std::istreambuf_iterator<char>());

    EncryptHelper decry;
    auto result = decry.decrypt(buffer, "");
    // assert(result.size() == buffer.size());

    model_type = model_type.substr(0, model_type.size() - sizeof(".encrypted") + 1) + ".buffer";
    return result;

  } else if (endswith(model_type, ".encrypted.buffer")) {
    EncryptHelper decry;
    auto result = decry.decrypt(data, "");
    if (!result.empty()) {
      model_type =
          model_type.substr(0, model_type.size() - sizeof(".encrypted.buffer") + 1) + ".buffer";
      return result;
    }
  } else
    return data;
  return data;
}

IPIPE_LOCAL void encrypt_buffer_to_file(const std::string& buffer, std::string out_file_path,
                                        std::string key) {
  if (!endswith(out_file_path, ".encrypted")) {
    SPDLOG_ERROR("out file name not end with .encrypted");
    throw std::runtime_error("out file name not end with .encrypted");
  }

  EncryptHelper decry;
  auto re = decry.encrypt(buffer, key);
  std::ofstream out_ff(out_file_path);
  out_ff << re;
  SPDLOG_INFO("{} saved", out_file_path);
  return;
}

IPIPE_LOCAL void encrypt_file_to_file(std::string file_path, std::string out_file_path,
                                      std::string key) {
  std::ifstream ff(file_path);
  if (!ff.good()) {
    SPDLOG_ERROR("open {} failed ", file_path);
    throw std::runtime_error("open failed: " + file_path);
  }
  std::string buffer((std::istreambuf_iterator<char>(ff)), std::istreambuf_iterator<char>());
  encrypt_buffer_to_file(buffer, out_file_path, key);
}

}  // namespace ipipe