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
#include <cassert>

#include <memory>
#include <string>
#include <vector>
#include "Backend.hpp"
#include "dict.hpp"
#include "event.hpp"
#include "MultipleInstances.hpp"
#include "params.hpp"
#include "reflect.h"
#include "threadsafe_queue_sized.hpp"
#include "time_utils.hpp"
#include "RangeMerger.hpp"
#include "RuningState.hpp"
#include "KVCacheManagerBase.hpp"
#include "base_logging.hpp"

// namespace kvcache {
// class KVCacheMaganer;
// }

namespace ipipe {

class RequestStates {
 public:
  RequestStates(size_t max_bs) : max_batch_size_(max_bs) {}
  struct RequestState {
   public:
    // int iter_index = 0;
    bool wait_for_schedule = true;
    size_t kvcache_seq_len = 1;
  };

  bool wait_decode_ready(int time_out);

  std::size_t size() {
    std::lock_guard<std::mutex> lock(mtx_);
    return request_states_.size();
  }

  void remove(const std::string& request_id) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      request_states_.erase(request_id);
    }
    cv_.notify_all();
  }

  // size_t get_iter_index(const std::string& request_id) {
  //   std::lock_guard<std::mutex> lock(mtx_);
  //   return request_states_.at(request_id).iter_index;
  // }

  size_t get_kvcache_seq_len(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    return request_states_.at(request_id).kvcache_seq_len;
  }

  bool has(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    return request_states_.find(request_id) != request_states_.end();
  }

  void set_wait(const std::string& request_id, size_t request_size) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      auto iter = request_states_.find(request_id);
      if (iter != request_states_.end()) {
        iter->second.wait_for_schedule = true;
        iter->second.kvcache_seq_len += request_size;
        assert(request_size == 1);
      } else {
        // (*request_states_)[request_id] = RequestState({0, true});
        request_states_.emplace(request_id, RequestState({true, request_size}));
      }
    }
    cv_.notify_all();
  }

  void set_unwait(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto iter = request_states_.find(request_id);
    if (iter != request_states_.end()) {
      iter->second.wait_for_schedule = false;
      // iter->second.iter_index += 1;
      // cv_.notify_all();
    }
  }
  // void notify_all() {
  //   std::lock_guard<std::mutex> lock(mtx_);
  //   cv_.notify_all();
  // }
  size_t get_max_batch_size() { return max_batch_size_; }

 private:
  std::unordered_map<std::string, RequestState> request_states_;
  mutable std::mutex mtx_;
  std::condition_variable cv_;
  size_t max_batch_size_;
};
// # cal_request_size_method = "AddRequestSizeTensor"

class Batching : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config);
  /**
   * @return UINT32_MAX.
   */
  virtual uint32_t max() const { return UINT32_MAX; };

  void forward(const std::vector<dict>& raw_inputs);

  bool contiguous_batch(dicts& input_data, const size_t input_data_size, const size_t new_pop,
                        dicts& redundant_data);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  ~Batching();
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
 protected:
#endif

  virtual void run();

 private:
  uint32_t max_batch_size_{1};
  std::thread thread_;
  ThreadSafeSizedQueue<dict> input_queue_;
  ThreadSafeSizedQueue<std::vector<dict>> batched_queue_;
  float batching_timeout_;

  std::unique_ptr<Params> params_;
  std::string node_name_;
  std::unique_ptr<Backend> backend_;
  std::unique_ptr<Backend> cal_request_size_method_;

  std::atomic_bool bThreadInited_{false};

  std::exception_ptr init_eptr_;

  // std::atomic<unsigned> count_{0};

  std::shared_ptr<RuningState> runing_state_;
  int contiguous_batching_{0};

  std::unique_ptr<RequestStates> request_states_;
  std::unique_ptr<kvcache::KVCacheManagerBase> kvcache_manager_;
};
}  // namespace ipipe
