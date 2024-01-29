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
#include <memory>
#include <string>
#include <vector>
#include "Backend.hpp"
#include "dict.hpp"
#include "event.hpp"
#include "params.hpp"
#include "reflect.h"
#include <list>
#include <ATen/ATen.h>

namespace ipipe {

struct DataWithNum {
  DataWithNum(int id_input, const std::vector<at::Tensor>& data_i) : id(id_input), data(data_i) {
    event = std::make_shared<SimpleEvents>();
  }
  std::vector<at::Tensor> data;
  int id;

  std::shared_ptr<SimpleEvents> event;
};
struct DataWithNumView {
  DataWithNumView(std::shared_ptr<DataWithNum> data) : data_(data) {}
  DataWithNumView(int id_input, const std::vector<at::Tensor>& data) {
    data_ = std::make_shared<DataWithNum>(id_input, data);
    length = data[0].size(0);
  }

  uint32_t start_offset{0};
  uint32_t length;
  std::shared_ptr<DataWithNum> data_;
  std::vector<at::Tensor> replay;
};

class TensorData {
 public:
  std::list<DataWithNumView> data_;
  std::unordered_map<int, std::vector<DataWithNumView>> borrowed_;
  // std::unordered_map<int, std::vector<DataWithNumView>> replay_;
  uint32_t sum_ = 0;
  uint32_t size() { return sum_; }
  void add(int id, const std::vector<at::Tensor>& data) {
    DataWithNumView data_id(id, data);
    data_.push_back(data_id);
    sum_ += data[0].size(0);
  }
  void reset() {
    // IPIPE_ASSERT(data_.empty());
    data_.clear();
    borrowed_.clear();
    sum_ = 0;
  }

  std::vector<std::vector<at::Tensor>> borrow(int id, uint32_t length) {  // can borrow enough data
    borrowed_[id] = std::vector<DataWithNumView>();
    std::vector<DataWithNumView>& result = borrowed_[id];
    for (auto iter = data_.begin(); iter != data_.end();) {
      if (length == 0) break;
      if (iter->length <= length) {
        sum_ -= iter->length;

        length -= iter->length;

        result.push_back(*iter);
        iter = data_.erase(iter);
      } else {
        DataWithNumView new_data = *iter;
        new_data.data_->event->task_add(1);
        new_data.length = length;
        sum_ -= length;
        result.push_back(new_data);

        iter->length -= length;
        iter->start_offset += length;
        break;
      }
    }
    std::vector<std::vector<at::Tensor>> final_data;
    for (const auto& item : result) {
      if (item.length == item.data_->data[0].size(0))
        final_data.push_back(item.data_->data);
      else {
        std::vector<at::Tensor> single_data;
        for (int j = 0; j < item.data_->data.size(); ++j) {
          single_data.push_back(
              item.data_->data[j].slice(0, item.start_offset, item.start_offset + item.length));
        }
        final_data.push_back(single_data);
      }
    }
    return final_data;
  }

  void set_replay(int id, const std::vector<std::vector<at::Tensor>>& replay) {
    for (std::size_t i = 0; i < replay.size(); ++i) {
      borrowed_[id][i].replay = replay[i];
      borrowed_[id][i].data_->event->notify_all();
    }
  }
  std::vector<std::vector<at::Tensor>> get_replay(int id) {
    std::vector<std::vector<at::Tensor>> final_data;
    std::vector<DataWithNumView> result;
    for (auto iter = borrowed_.begin(); iter != borrowed_.end(); ++iter) {
      for (auto iter_inner = iter->second.begin(); iter_inner != iter->second.end(); ++iter_inner) {
        if (iter_inner->data_->id == id) {
          result.push_back(*iter_inner);
          break;
        }
      }
    }
    // sort result by start_offset:
    std::sort(result.begin(), result.end(), [](const DataWithNumView& a, const DataWithNumView& b) {
      return a.start_offset < b.start_offset;
    });
    for (const auto& item : result) {
      item.data_->event->Wait();
      final_data.push_back(item.replay);
    }
    return final_data;
  }

 private:
  std::mutex lock_;
  /* data */
};

class BorrowReplay : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) override;

  virtual uint32_t max() const { return 1; };

  void forward(const std::vector<dict>& raw_inputs);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~BorrowReplay();
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
 protected:
#endif

 private:
  std::unique_ptr<Params> params_;

  TensorData pool_;

  uint32_t max_batch_size_;
  std::mutex lock_;
};
}  // namespace ipipe
