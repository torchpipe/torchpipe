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

#ifdef USE_DECRYPT
#include "decrypt.hpp"
#include "spdlog/spdlog.h"

namespace ipipe {

void handleOpenSSLErrors(void) {
  ERR_print_errors_fp(stderr);
  SPDLOG_ERROR("error: decrypt model failed, please examine your key!! ");
  // abort();
}

bool decrypt(unsigned char* ciphertext, int ciphertext_len, unsigned char* key, unsigned char* iv,
             std::vector<char>& decrypt_data) {
  EVP_CIPHER_CTX* ctx;
  int len;
  int plaintext_len;

  auto data = std::vector<unsigned char>(ciphertext_len);
  unsigned char* plaintext = data.data() bzero(plaintext, ciphertext_len);

  if (!(ctx = EVP_CIPHER_CTX_new())) {
    handleOpenSSLErrors();
    return false;
  }

  if (1 != EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv)) {
    handleOpenSSLErrors();
    return false;
  }

  EVP_CIPHER_CTX_set_key_length(ctx, EVP_MAX_KEY_LENGTH);

  if (1 != EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len)) {
    handleOpenSSLErrors();
    return false;
  }

  plaintext_len = len;

  if (1 != EVP_DecryptFinal_ex(ctx, plaintext + len, &len)) {
    handleOpenSSLErrors();
    return false;
  }

  plaintext_len += len;
  /* Add the null terminator */
  plaintext[plaintext_len] = 0;

  /* Clean up */
  EVP_CIPHER_CTX_free(ctx);
  decrypt_data.resize(plaintext_len);

  memcpy(&decrypt_data[0], plaintext,
         plaintext_len);  // plaintext -> decrypt_data(vector)

  return true;
}

void initAES(const std::string& pass, unsigned char* salt, unsigned char* key, unsigned char* iv,
             int pbkdf2 = 0) {
  bzero(key, sizeof(key));
  bzero(iv, sizeof(iv));
  if (pbkdf2 == 0) {
    EVP_BytesToKey(EVP_aes_256_cbc(), EVP_sha1(), salt, (unsigned char*)pass.c_str(), pass.length(),
                   1, key, iv);
  } else if (pbkdf2 == 1) {
    int iter = 1;
    unsigned char tmpkeyiv[EVP_MAX_KEY_LENGTH + EVP_MAX_IV_LENGTH];
    int iklen = 32;  // EVP_CIPHER_get_key_length(cipher);
    int ivlen = 16;  // EVP_CIPHER_get_iv_length(cipher);
    /* not needed if HASH_UPDATE() is fixed : */
    int islen = (salt != NULL ? sizeof(salt) : 0);
    PKCS5_PBKDF2_HMAC((char*)pass.c_str(), pass.length(), salt, islen, iter, EVP_sha1(),
                      iklen + ivlen, tmpkeyiv);
    /* split and move data back to global buffer */
    memcpy(key, tmpkeyiv, iklen);
    memcpy(iv, tmpkeyiv + iklen, ivlen);
  } else {
    SPDLOG_ERROR(" error :pbkdf2 only support 0 or 1. ");
  }
}

bool aes_256_decrypt_from_cache(unsigned char* ciphertext, size_t cipher_len,
                                std::vector<char>& decrypt_data, int pbkdf2,
                                const std::string& password) {
  ERR_load_crypto_strings();

  unsigned char salt[8];
  unsigned char key[32];
  unsigned char iv[32];

  if (strncmp((const char*)ciphertext, "Salted__", 8) == 0) {
    memcpy(salt, &ciphertext[8], 8);
    ciphertext += 16;
    cipher_len -= 16;
  }

  initAES(password, salt, key, iv, pbkdf2);

  bool succ = decrypt(ciphertext, cipher_len, key, iv, decrypt_data);
  EVP_cleanup();
  ERR_free_strings();
  return succ;
}

/*
 *  解密函数，输入加密模型的path，实现解密
 *
 *
 */
bool aes_256_decrypt_from_path(const std::string& path, std::vector<char>& decrypt_data, int pbkdf2,
                               const std::string& password) {
  SPDLOG_INFO("Decrypting model: " + path);
  std::vector<char> ori_data;
  size_t size = 0;
  std::ifstream ifile(path, std::ios::binary);
  if (ifile.good()) {
    ifile.seekg(0, ifile.end);
    size = ifile.tellg();
    ifile.seekg(0, ifile.beg);
    ori_data.resize(size);
    ifile.read(ori_data.data(), size);
    ifile.close();
  } else {
    SPDLOG_ERROR(path + " not exists.\n\n");
    return false;
  }
  SPDLOG_INFO("Model Size: " + std::to_string(size));
  unsigned char* ciphertext = reinterpret_cast<unsigned char*>(ori_data.data());
  bool succ = aes_256_decrypt_from_cache(ciphertext, size, decrypt_data, pbkdf2, password);

  return succ;
}

}  // namespace ipipe

#endif