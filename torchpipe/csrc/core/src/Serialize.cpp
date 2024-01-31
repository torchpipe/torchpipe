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

#include "Serialize.hpp"
#include <string>
#include <unordered_map>
#include "ipipe_common.hpp"
// #if __cplusplus > 201703L
// #include <variant>
// #endif
#include <cstddef>
#include <cstring>
// #if __cplusplus > 201703L
// #endif

#ifdef WITH_OPENCV
#include <opencv2/core.hpp>
#endif

/** supported types:
 * std::vector<char>  0
 * std::vector<uint8> 1
 * std::string        2
 * cv::Mat            3
 */

#define HEADER_TAG 9789
namespace {

struct alignas(uint32_t) DictHeader {
  uint32_t tag = HEADER_TAG;
  unsigned char reserved[4];
  uint32_t number;  // len1;key1;len_value1;value1   ;len2;key2;len3;key3
  uint32_t data_len;
};
}  // namespace
namespace ipipe {

template <class T>
bool get_data(const any& data, unsigned char index, std::vector<char>& result) {
  const auto* true_data = any_cast<T>(&data);
  if (!true_data) return false;
  result.resize(true_data->size() + 4);
  result[3] = index;
  std::memcpy(result.data() + 4, true_data->data(), true_data->size());
  return true;
}

template <class T>
bool get_data(const std::string& data, unsigned char index, std::vector<char>& result) {
  result.resize(data.size() + 4);
  result[3] = index;
  std::memcpy(result.data() + 4, data.data(), data.size());
  return true;
}

bool serialize(const any& data, std::vector<char>& result) {
  if (data.type() == typeid(std::vector<char>)) {
    return get_data<std::vector<char>>(data, 0, result);
  } else if (data.type() == typeid(std::vector<unsigned char>)) {
    return get_data<std::vector<unsigned char>>(data, 1, result);
  } else if (data.type() == typeid(std::string)) {
    return get_data<std::string>(data, 2, result);
  }
#ifdef WITH_OPENCV
  else if (data.type() == typeid(cv::Mat)) {
    cv::Mat d = *any_cast<cv::Mat>(&data);
    if (!d.isContinuous()) d = d.clone();
    // IPIPE_ASSERT(d);
    result.resize(d.rows * d.cols * d.channels() * d.elemSize1() + 4 + 4 * sizeof(uint32_t));
    result[3] = 3;
    uint32_t* phwc = (uint32_t*)(result.data() + 4);
    phwc[0] = d.rows;
    phwc[1] = d.cols;
    phwc[2] = d.channels();
    phwc[3] = d.elemSize1();
    std::memcpy(result.data() + 4 + 4 * sizeof(uint32_t), d.data,
                d.rows * d.cols * d.channels() * d.elemSize1());
    return true;

  }
#endif
  else {
    return false;
  }
}

bool serialize(const std::string& data, std::vector<char>& result) {
  return get_data<std::string>(data, 2, result);
}

any deserialize(const char* start, uint32_t len, bool& success) {
  switch (start[3]) {
    case 0:
      IPIPE_ASSERT(len >= 4);
      len -= 4;
      success = true;
      return std::vector<char>(start + 4, start + len);
    case 1:
      IPIPE_ASSERT(len >= 4);
      len -= 4;
      success = true;
      return std::vector<unsigned char>(start + 4, start + len);
    case 2:
      IPIPE_ASSERT(len >= 4);
      len -= 4;
      success = true;
      return std::string(start + 4, len);
#ifdef WITH_OPENCV
    case 3: {
      IPIPE_ASSERT(len >= 4 + 4 * sizeof(uint32_t));
      len -= 4 + 4 * sizeof(uint32_t);
      const uint32_t* phwc = (const uint32_t*)(start + 4);
      IPIPE_ASSERT(phwc[0] > 0 && phwc[1] > 0 && phwc[2] > 0 && (phwc[3] == 1 || phwc[3] == 4));
      success = true;
      if (phwc[3] == 1) {
        cv::Mat mat(phwc[0], phwc[1], CV_8UC(phwc[2]), (char*)start + 4 + 4 * sizeof(uint32_t));
        return mat;
      } else {
        cv::Mat mat(phwc[0], phwc[1], CV_32FC(phwc[2]), (char*)start + 4 + 4 * sizeof(uint32_t));
        return mat;
      }
      break;
    }
#endif

    default:
      throw std::runtime_error("deserialize failed");
  }
  return any();
}

std::vector<char> serialize(const std::unordered_map<std::string, std::string>& map_data) {
  const auto* data = &map_data;
  std::unordered_map<std::string, std::vector<char>> results;
  uint32_t total = sizeof(DictHeader);
  for (const auto& item : *data) {
    std::vector<char> result;
    if (serialize(item.second, result)) {
      total += item.first.size();
      total += result.size();
      results[item.first] = std::move(result);
    }
  }
  total += 2 * sizeof(uint32_t) * results.size();

  DictHeader head;
  head.number = results.size();
  head.data_len = total - sizeof(DictHeader);

  std::vector<char> result;
  result.resize(total);
  // result.resize(sizeof(DictHeader) + 2 * sizeof(uint32_t) * results.size());
  std::memcpy(result.data(), &head, sizeof(DictHeader));
  uint32_t* index = reinterpret_cast<uint32_t*>(result.data() + sizeof(DictHeader));
  char* value = result.data() + sizeof(DictHeader) + 2 * sizeof(uint32_t) * results.size();
  for (const auto& item : results) {
    const auto len_key = item.first.size();
    const auto len_value = item.second.size();
    index[0] = len_key;
    index[1] = len_value;
    index += 2;

    std::memcpy(value, item.first.data(), len_key);
    value += len_key;
    std::memcpy(value, item.second.data(), len_value);
    value += len_value;
  }
  return result;
}

std::vector<char> serialize(dict data) {
  IPIPE_ASSERT(data && data->find(TASK_DATA_KEY) != data->end());
  std::unordered_map<std::string, std::vector<char>> results;
  uint32_t total = sizeof(DictHeader);
  for (const auto& item : *data) {
    std::vector<char> result;
    if (serialize(item.second, result)) {
      total += item.first.size();
      total += result.size();
      results[item.first] = std::move(result);
    }
  }
  total += 2 * sizeof(uint32_t) * results.size();

  DictHeader head;
  head.number = results.size();
  head.data_len = total - sizeof(DictHeader);

  std::vector<char> result;
  result.resize(total);
  // result.resize(sizeof(DictHeader) + 2 * sizeof(uint32_t) * results.size());
  std::memcpy(result.data(), &head, sizeof(DictHeader));
  uint32_t* index = reinterpret_cast<uint32_t*>(result.data() + sizeof(DictHeader));
  char* value = result.data() + sizeof(DictHeader) + 2 * sizeof(uint32_t) * results.size();
  for (const auto& item : results) {
    const auto len_key = item.first.size();
    const auto len_value = item.second.size();
    index[0] = len_key;
    index[1] = len_value;
    index += 2;

    std::memcpy(value, item.first.data(), len_key);
    value += len_key;
    std::memcpy(value, item.second.data(), len_value);
    value += len_value;
  }
  return result;
}

void deserialize(const std::vector<char>& data,
                 std::unordered_map<std::string, std::string>& result) {
  IPIPE_ASSERT(data.size() >= sizeof(DictHeader));
  const DictHeader* head = reinterpret_cast<const DictHeader*>(data.data());
  IPIPE_ASSERT(head->tag == HEADER_TAG && head->data_len + sizeof(DictHeader) == data.size());
  const uint32_t* index = reinterpret_cast<const uint32_t*>(data.size() + sizeof(DictHeader));
  const char* value = data.data() + sizeof(DictHeader) + 2 * sizeof(uint32_t) * head->number;
  for (std::size_t i = 0; i < head->number; ++i) {
    const auto len_key = index[0];
    const auto len_value = index[1];
    index += 2;
    IPIPE_ASSERT(len_value >= 4);

    result[std::string(value, len_key)] = std::string(value + len_key + 4, len_value - 4);
    value += len_key + len_value;
  }

  return;
}

dict deserialize(const std::vector<char>& data) {
  dict result = make_dict();
  IPIPE_ASSERT(data.size() >= sizeof(DictHeader));
  const DictHeader* head = reinterpret_cast<const DictHeader*>(data.data());
  IPIPE_ASSERT(head->tag == HEADER_TAG && head->data_len + sizeof(DictHeader) == data.size());
  const uint32_t* index = reinterpret_cast<const uint32_t*>(data.size() + sizeof(DictHeader));
  const char* value = data.data() + sizeof(DictHeader) + 2 * sizeof(uint32_t) * head->number;
  auto& map_data = *result;
  for (std::size_t i = 0; i < head->number; ++i) {
    const auto len_key = index[0];
    const auto len_value = index[1];
    index += 2;
    bool succ = false;
    auto re = deserialize(value + len_key, len_value, succ);
    IPIPE_ASSERT(succ);
    map_data[std::string(value, len_key)] = re;
    value += len_key + len_value;
  }

  return result;
}
}  // namespace ipipe
