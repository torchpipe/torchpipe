// Copyright 2021-2026 NetEase.
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

#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string_view>
#include <vector>

#include "encrypt.hpp"
#include "aes.h"
#ifndef TORCHPIPE_TENSORRT_KEY_HEX
#error "TORCHPIPE_TENSORRT_KEY_HEX is not defined. Build torchpipe_tensorrt via torchpipe.utils._build_trt."
#endif
#define TORCHPIPE_STRINGIFY_IMPL(value) #value
#define TORCHPIPE_STRINGIFY(value) TORCHPIPE_STRINGIFY_IMPL(value)


namespace {

template <typename T, std::size_t N>
constexpr uint32_t array_sum(T (&array)[N]) {
  uint32_t sum = 0;
  for (std::size_t i = 0; i < N; i++) {
    sum += array[i];
  }
  return sum;
};

template <typename T, std::size_t N>
uint32_t array_sum(const std::array<T, N>& array) {
  uint32_t sum = 0;
  for (const auto value : array) {
    sum += value;
  }
  return sum;
};

uint32_t array_sum(const std::string& in) {
  uint32_t sum = 0;
  for (std::size_t i = 0; i < in.size(); i++) {
    sum += static_cast<unsigned char>(in[i]);
  }
  return sum;
};

uint32_t array_sum(const std::vector<unsigned char>& in) {
  uint32_t sum = 0;
  for (unsigned char value : in) {
    sum += value;
  }
  return sum;
};

using Sha256Digest = std::array<unsigned char, 32>;

uint32_t rotr(uint32_t value, uint32_t bits) {
  return (value >> bits) | (value << (32 - bits));
}

Sha256Digest sha256_digest(std::string_view input) {
  static constexpr std::array<uint32_t, 64> kSha256Constants = {
      0x428a2f98,
      0x71374491,
      0xb5c0fbcf,
      0xe9b5dba5,
      0x3956c25b,
      0x59f111f1,
      0x923f82a4,
      0xab1c5ed5,
      0xd807aa98,
      0x12835b01,
      0x243185be,
      0x550c7dc3,
      0x72be5d74,
      0x80deb1fe,
      0x9bdc06a7,
      0xc19bf174,
      0xe49b69c1,
      0xefbe4786,
      0x0fc19dc6,
      0x240ca1cc,
      0x2de92c6f,
      0x4a7484aa,
      0x5cb0a9dc,
      0x76f988da,
      0x983e5152,
      0xa831c66d,
      0xb00327c8,
      0xbf597fc7,
      0xc6e00bf3,
      0xd5a79147,
      0x06ca6351,
      0x14292967,
      0x27b70a85,
      0x2e1b2138,
      0x4d2c6dfc,
      0x53380d13,
      0x650a7354,
      0x766a0abb,
      0x81c2c92e,
      0x92722c85,
      0xa2bfe8a1,
      0xa81a664b,
      0xc24b8b70,
      0xc76c51a3,
      0xd192e819,
      0xd6990624,
      0xf40e3585,
      0x106aa070,
      0x19a4c116,
      0x1e376c08,
      0x2748774c,
      0x34b0bcb5,
      0x391c0cb3,
      0x4ed8aa4a,
      0x5b9cca4f,
      0x682e6ff3,
      0x748f82ee,
      0x78a5636f,
      0x84c87814,
      0x8cc70208,
      0x90befffa,
      0xa4506ceb,
      0xbef9a3f7,
      0xc67178f2,
  };
  std::array<uint32_t, 8> hash = {
      0x6a09e667,
      0xbb67ae85,
      0x3c6ef372,
      0xa54ff53a,
      0x510e527f,
      0x9b05688c,
      0x1f83d9ab,
      0x5be0cd19,
  };

  std::vector<unsigned char> message(input.begin(), input.end());
  const uint64_t bit_length = static_cast<uint64_t>(message.size()) * 8;
  message.push_back(0x80);
  while (message.size() % 64 != 56) {
    message.push_back(0);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(static_cast<unsigned char>((bit_length >> shift) & 0xff));
  }

  for (std::size_t offset = 0; offset < message.size(); offset += 64) {
    std::array<uint32_t, 64> words {};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t index = offset + i * 4;
      words[i] = (static_cast<uint32_t>(message[index]) << 24) |
          (static_cast<uint32_t>(message[index + 1]) << 16) |
          (static_cast<uint32_t>(message[index + 2]) << 8) |
          static_cast<uint32_t>(message[index + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      const uint32_t s0 =
          rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^
          (words[i - 15] >> 3);
      const uint32_t s1 =
          rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^
          (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = hash[0];
    uint32_t b = hash[1];
    uint32_t c = hash[2];
    uint32_t d = hash[3];
    uint32_t e = hash[4];
    uint32_t f = hash[5];
    uint32_t g = hash[6];
    uint32_t h = hash[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const uint32_t sigma1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t temp1 =
          h + sigma1 + choose + kSha256Constants[i] + words[i];
      const uint32_t sigma0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = sigma0 + majority;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }

  Sha256Digest digest {};
  for (std::size_t i = 0; i < hash.size(); ++i) {
    digest[i * 4] = static_cast<unsigned char>((hash[i] >> 24) & 0xff);
    digest[i * 4 + 1] = static_cast<unsigned char>((hash[i] >> 16) & 0xff);
    digest[i * 4 + 2] = static_cast<unsigned char>((hash[i] >> 8) & 0xff);
    digest[i * 4 + 3] = static_cast<unsigned char>(hash[i] & 0xff);
  }
  return digest;
}

unsigned char decode_hex_nibble(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<unsigned char>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<unsigned char>(10 + value - 'a');
  }
  if (value >= 'A' && value <= 'F') {
    return static_cast<unsigned char>(10 + value - 'A');
  }
  throw std::runtime_error("TORCHPIPE_TENSORRT_KEY_HEX contains non-hex characters.");
}

Sha256Digest parse_key_hex(std::string_view key_hex) {
  if (key_hex.size() != 64) {
    throw std::runtime_error("TORCHPIPE_TENSORRT_KEY_HEX must contain 64 hex characters.");
  }
  Sha256Digest key_bytes {};
  for (std::size_t i = 0; i < key_bytes.size(); ++i) {
    const auto high = decode_hex_nibble(key_hex[i * 2]);
    const auto low = decode_hex_nibble(key_hex[i * 2 + 1]);
    key_bytes[i] = static_cast<unsigned char>((high << 4) | low);
  }
  return key_bytes;
}

const Sha256Digest& get_compiled_key_bytes() {
  static const Sha256Digest key_bytes =
      parse_key_hex(TORCHPIPE_STRINGIFY(TORCHPIPE_TENSORRT_KEY_HEX));
  return key_bytes;
}

struct KeyMaterial {
  AESKeyLength key_length;
  std::vector<unsigned char> key_bytes;
  uint32_t tag_sum;
};

KeyMaterial get_primary_key_material(const std::string& key) {
  if (key.empty()) {
    const auto& key_bytes = get_compiled_key_bytes();
    return {
        AESKeyLength::AES_256,
        std::vector<unsigned char>(key_bytes.begin(), key_bytes.end()),
        array_sum(key_bytes),
    };
  }

  const auto digest = sha256_digest(key);
  return {
      AESKeyLength::AES_256,
      std::vector<unsigned char>(digest.begin(), digest.end()),
      array_sum(digest),
  };
}



class OMNI_LOCAL EncryptHelper {
 public:
  std::string encrypt(const char* buffer, size_t buffer_len, std::string key) {
    const auto key_material = get_primary_key_material(key);
    PaddingHead header;
    header.tag = get_torchpipe_tag(key_material);
    int total_len =
        4 * 16 - buffer_len % 16 + buffer_len + (sizeof(PaddingHead) % 16) * 16;
    header.data_start = total_len - buffer_len;

    header.data_len = buffer_len;

    std::vector<unsigned char> tmp_vector(total_len);
    assert(tmp_vector.size() % 16 == 0);
    std::memcpy(tmp_vector.data(), &header, sizeof(PaddingHead));
    if (buffer_len > 0) {
      std::memcpy(tmp_vector.data() + header.data_start, buffer, buffer_len);
    }

    AES aes(key_material.key_length);
    auto re = aes.EncryptECB(tmp_vector, key_material.key_bytes);

    return std::string(re.begin(), re.end());
  }

  std::vector<unsigned char> decrypt(
      const std::vector<unsigned char>& buffer,
      std::string key) {
    const auto key_material = get_primary_key_material(key);
    return decrypt_with_key_material(buffer, key_material);
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
  std::vector<unsigned char> decrypt_with_key_material(
      const std::vector<unsigned char>& buffer,
      const KeyMaterial& key_material) {
    AES aes(key_material.key_length);
    auto result = aes.DecryptECB(buffer, key_material.key_bytes);
    assert(result.size() % 16 == 0);

    return get_data(result, get_torchpipe_tag(key_material));
  }

  std::vector<unsigned char> get_data(
      const std::vector<unsigned char>& data,
      uint32_t expected_tag) {
    if (data.size() < sizeof(PaddingHead)) {
      throw std::runtime_error("data.size() < sizeof(head) ");
    }
    const PaddingHead* header =
        reinterpret_cast<const PaddingHead*>(data.data());
    const auto data_start = static_cast<std::size_t>(header->data_start);
    const auto data_len = static_cast<std::size_t>(header->data_len);
    const auto data_end = data_start + data_len;
    if (header->tag != expected_tag || data_start > data.size() ||
        data_end != data.size()) {
      throw std::runtime_error("DECRYPT: tag or version not match.");
    }
    const auto* p_start = data.data() + data_start;
    return std::vector<unsigned char>(p_start, p_start + data_len);
  }

  uint32_t get_torchpipe_tag(const KeyMaterial& key_material) {
    return 5432023 + sizeof(PaddingHead) * key_material.tag_sum;
  }
};

std::vector<unsigned char> read_file_binary(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("open failed: " + path);
  }

  file.seekg(0, std::ios::end);
  const auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (file_size == std::ifstream::pos_type(-1)) {
    throw std::runtime_error("tellg failed: " + path);
  }

  std::vector<unsigned char> buffer(static_cast<size_t>(file_size));
  if (!buffer.empty()) {
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!file.good()) {
      throw std::runtime_error("read failed: " + path);
    }
  }
  return buffer;
}

void write_error_message(
    char* error_message,
    size_t error_message_size,
    const std::string& message) {
  if (!error_message || error_message_size == 0) {
    return;
  }
  std::snprintf(error_message, error_message_size, "%s", message.c_str());
}
} // namespace

namespace torchpipe {
 
OMNI_LOCAL std::vector<unsigned char> decrypt_file(std::string path) {
  const auto encrypted = read_file_binary(path);

  EncryptHelper decry;
  std::vector<unsigned char> result = decry.decrypt(encrypted, "");

  return result;
}

OMNI_LOCAL void encrypt_file_to_file(
    const std::string& file_path,
    const std::string& out_file_path) {
  const auto data = read_file_binary(file_path);
  encrypt2file(
      reinterpret_cast<const char*>(data.data()), data.size(), out_file_path);
}

OMNI_LOCAL void encrypt2file(
    const char* data,
    size_t data_len,
    std::string out_file_path) {


  EncryptHelper decry;
  auto re = decry.encrypt(data, data_len, "");
  std::ofstream out_ff(out_file_path, std::ios::binary);
  if (!out_ff.good()) {
    throw std::runtime_error("open failed: " + out_file_path);
  }
  out_ff << re;
  return;

}

} // namespace torchpipe

extern "C" TORCHPIPE_TENSORRT_EXPORT int torchpipe_encrypt_file(
    const char* input_path,
    const char* output_path,
    char* error_message,
    size_t error_message_size) {
  try {
    if (!input_path || !output_path) {
      write_error_message(error_message, error_message_size, "input_path or output_path is null");
      return -1;
    }
    torchpipe::encrypt_file_to_file(input_path, output_path);
    write_error_message(error_message, error_message_size, "");
    return 0;
  } catch (const std::exception& e) {
    write_error_message(error_message, error_message_size, e.what());
    return -1;
  } catch (...) {
    write_error_message(error_message, error_message_size, "unknown error");
    return -1;
  }
}
