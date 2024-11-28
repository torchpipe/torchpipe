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

#include "Batching.hpp"
#include "time_utils.hpp"
#include <chrono>
#include <functional>
#include <sstream>
#include <thread>
#include <tuple>
#include "base_logging.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "Instances.hpp"
#include "time_utils.hpp"
#include "threadsafe_kv_storage.hpp"
// #include "KVCacheManagerBase.hpp"
#include "config_parser.hpp"
#include "sampling_params.h"
#include "base_logging.hpp"

namespace ipipe {

namespace {
// 定义 get_request_ids 函数
std::set<std::string> get_request_ids(const std::vector<dict>& input_data) {
  std::set<std::string> request_ids;
  for (const auto& request : input_data) {
    auto iter = request->find("request_id");
    if (iter == request->end()) {
      SPDLOG_ERROR("request_id not found in contiguous batching mode");
      continue;
    }
    std::string* request_id = any_cast<std::string>(&iter->second);
    request_ids.insert(*request_id);
  }
  return request_ids;
}

// void read_kvcache(kvcache::KVCacheConfig& kv_config) {
//   kv_config.device_id = get_registered_config("kvcache", "device_id", -1);
//   kv_config.elemsize = get_registered_config("kvcache", "elemsize", 2);

//   kv_config.layer_num = get_registered_config("kvcache", "layer_num", 32);
//   kv_config.max_seq_len = get_registered_config("kvcache", "max_seq_len", 2048);
//   kv_config.hidden_size = get_registered_config("kvcache", "hidden_size", 4096);
//   kv_config.num_heads = get_registered_config("kvcache", "num_heads", 32);
//   kv_config.granularitySize = get_registered_config("kvcache", "granularitySize", 2 * 1024 *
//   1024); kv_config.max_concurrent_requests =
//       get_registered_config("kvcache", "max_concurrent_requests", 256);
//   kv_config.reserve_prefill = get_registered_config("kvcache", "reserve_prefill", 64);
//   kv_config.reserve_decode = get_registered_config("kvcache", "reserve_decode", 1);
// }
}  // namespace

bool RequestStates::wait_all_ready(int time_out) {
  std::unique_lock<std::mutex> lock(mtx_);
  return cv_.wait_for(lock, std::chrono::milliseconds(time_out), [this]() {
    if (!all_ready_) return false;
    for (auto iter = request_states_.begin(); iter != request_states_.end(); ++iter) {
      if (!iter->second.wait_for_schedule) {
        // SPDLOG_INFO("not ready: {}", iter->first);
        return false;
      }
    }
    return true;
  });
}

Batching::~Batching() {
  bThreadInited_.store(false);
  if (!input_queue_.empty()) {
    SPDLOG_ERROR("!input_queue_.empty()");
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void Batching::forward(const std::vector<dict>& raw_inputs) {
  if (cal_request_size_method_) {
    // if (!bThreadInited_.load()) {
    //   SPDLOG_ERROR("cal_request_size_method_ is not supported when no batching needed");
    //   abort();
    // }
    for (const auto& item : raw_inputs) cal_request_size_method_->forward({item});
  }
  std::vector<std::shared_ptr<SimpleEvents>> events;  // 注意，
  // 事件需要提前准备好，不可运行时从map获得，容易造成多线程问题

  for (auto raw_input : raw_inputs) {
    std::shared_ptr<RuningStateMonitor> guard_state =
        std::make_shared<RuningStateMonitor>(runing_state_, 1);
    assert(guard_state);
    auto& map_data = *raw_input;
    map_data.erase(TASK_RESULT_KEY);

    auto iter = raw_input->find(TASK_EVENT_KEY);
    if (iter == raw_input->end()) {
      auto event = make_event();
      events.emplace_back(event);
      event->add_const_callback([guard_state]() { guard_state->del(); });
      if (cal_request_size_method_) {
        auto* data = raw_input.get();
        event->add_const_callback([data]() { data->erase(TASK_REQUEST_SIZE_KEY); });
      }
      map_data[TASK_EVENT_KEY] = event;
    } else {
      events.emplace_back(nullptr);

      std::shared_ptr<SimpleEvents> ev = any_cast<std::shared_ptr<SimpleEvents>>(iter->second);
      ev->add_const_callback([guard_state]() { guard_state->del(); });
      if (cal_request_size_method_) {
        auto* data = raw_input.get();
        ev->add_const_callback([data]() { data->erase(TASK_REQUEST_SIZE_KEY); });
      }
    }
  }

  assert(events.size() == raw_inputs.size());

  {
    // 注意：资源所有权问题， 从此刻起 对 raw_input 没有读写权限，
    // 除非event通知

    if (!bThreadInited_.load()) {
      for (auto raw_input : raw_inputs) {
        backend_->forward({raw_input});  // 异步调用, bs=1
      }
    } else {
      std::vector<size_t> sizes;
      for (const auto& item : raw_inputs) {
        const auto item_size = get_request_size(item);
        IPIPE_ASSERT(item_size > 0);
        // SPDLOG_DEBUG("item_size={} max_batch_size={}", item_size, max_batch_size_);
        IPIPE_ASSERT(item_size <= max_batch_size_);
        sizes.push_back(item_size);
      }

      if (request_states_) {
        // size_t num_prefill = 0;
        request_states_->set_unready();
        for (size_t i = 0; i < raw_inputs.size(); ++i) {
          const auto& request = raw_inputs[i];
          const auto size = sizes[i];
          auto iter = request->find("request_id");
          if (iter == request->end()) {
            SPDLOG_ERROR("request_id not found in contiguous batching mode");
            continue;
          }

          std::string* request_id = any_cast<std::string>(&iter->second);
          // request_states_->set_unready();
          // if (!request_states_->has(*request_id)) {
          //   kvcache::KVCacheAllocParams request_params;
          //   auto& storage = ThreadSafeKVStorage::getInstance().get(*request_id);

          //   const llm::SamplingParams* samp =
          //       any_cast<llm::SamplingParams>(&storage.get("sampling_params"));

          //   request_params.max_new_tokens = samp->max_tokens;
          //   request_params.request_id = *request_id;
          //   request_params.kvcache_seq_len = size;
          //   // request_params.max_new_tokens;
          //   SPDLOG_WARN("contiguous_batching: alloc_reqid: request_id={}, kvcache_seq_len={}",
          //               request_params.request_id, request_params.kvcache_seq_len);
          //   kvcache_manager_->alloc_reqid(request_params);
          //   request_states_.emplace(request_id, RequestState({true, request_size}));
          //   // num_prefill++;
          // }
          request_states_->set_ready(*request_id, size);
          // SPDLOG_INFO("contiguous_batching: set_ready {}", *request_id);
        }
        input_queue_.Push(raw_inputs, sizes);
        request_states_->notify();
      }
    }
  }

  for (std::size_t i = 0; i < raw_inputs.size(); ++i) {
    // 当阻塞式调用时 todo  非阻塞调用
    if (events[i]) {
      events[i]->Wait();
      // 重新获得资源所有权

      raw_inputs[i]->erase(TASK_EVENT_KEY);
    } else {
      // 无资源所有权
      continue;
    }
  }

  return;
};

bool Batching::init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
  params_ =
      std::unique_ptr<Params>(new Params({{"multiple_instances", ""},
                                          {"batching_timeout", "1"},
                                          {"cal_request_size_method", ""},  // AddRequestSizeTensor
                                          {"node_name", ""},
                                          {"contiguous_batching", "0"}},
                                         {}, {}, {}));
  if (config.empty()) {
    SPDLOG_ERROR("empty config. Only support single-node configuration.");
    return false;
  }
  if (!params_->init(config)) return false;
  auto batching_timeouts = str_split(params_->at("batching_timeout"), '&');
  batching_timeout_ = 0;
  for (const auto& item : batching_timeouts) {
    batching_timeout_ = std::max(batching_timeout_, std::stof(item));
  }

  node_name_ = params_->at("node_name");

  if (params_->at("multiple_instances").empty()) {
    // backend_ = std::make_unique<RangeMerger>();
    backend_ = std::make_unique<MultiInstances>();
  } else {
    backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("multiple_instances")));
  }
  // batched_queue_ = std::make_unique<ThreadSafeSizedQueue<std::vector<dict>>>();
  (*dict_config)["_batched_queue"] = &batched_queue_;

  if (!params_->at("cal_request_size_method").empty()) {
    cal_request_size_method_ =
        std::unique_ptr<Backend>(IPIPE_CREATE(Backend, params_->at("cal_request_size_method")));
    IPIPE_ASSERT(cal_request_size_method_ && cal_request_size_method_->init(config, dict_config));
  }
  if (!backend_ || !backend_->init(config, dict_config)) return false;
  runing_state_ = std::make_shared<RuningState>();
  {
    max_batch_size_ = backend_->max();
    if (max_batch_size_ == UINT32_MAX) {
      SPDLOG_WARN(node_name_ + ": max() == UINT32_MAX");
    }

    if (max_batch_size_ != 1 && batching_timeout_ > 0) {
      bThreadInited_.store(true);
      thread_ = std::thread(&Batching::run, this);
    } else if (max_batch_size_ != 1 && batching_timeout_ == 0) {
      SPDLOG_WARN(
          "{}: Batching will not be enabled as batching_timeout is set to 0. Even though "
          "max_batch_size is greater than 1, multiple requests coming in simultaneously will not "
          "be batched together.",
          node_name_);
    }
    SPDLOG_INFO("{}: max_batch_size={}, batching_timeout={}", node_name_, max_batch_size_,
                batching_timeout_);
  }

  contiguous_batching_ = std::stoi(params_->at("contiguous_batching"));
  if (contiguous_batching_) {
    IPIPE_ASSERT(bThreadInited_.load());
    IPIPE_ASSERT(!node_name_.empty(),
                 "node_name should not be empty when contiguous_batching is enabled");
    SPDLOG_INFO("contiguous batching is enabled");

    request_states_ = std::make_unique<RequestStates>();
    original_max_batch_size_ = max_batch_size_;
    // max_batch_size_ = UINT32_MAX;
    auto& storage = ThreadSafeKVStorage::getInstance(ThreadSafeKVStorage::POOL::REQUEST_ID);
    storage.add_remove_callback([this](const std::string& req) {
      SPDLOG_INFO("remove request_id: {} from callback", req);
      // order:
      // kvcache_manager_->free_reqid(req);
      request_states_->remove(req);
    });
    // kvcache_manager_ = std::unique_ptr<kvcache::KVCacheManagerBase>(
    //     IPIPE_CREATE(kvcache::KVCacheManagerBase, "KVCacheManager"));
    // kvcache::KVCacheConfig kv_config;
    // read_kvcache(kv_config);
    // IPIPE_ASSERT(kv_config.max_batch_size <= 0 ||
    //              kv_config.max_batch_size == original_max_batch_size_);
    // kv_config.max_batch_size = original_max_batch_size_;

    // kvcache_manager_->init(kv_config);

    // ThreadSafeKVStorage::getInstance(ThreadSafeKVStorage::POOL::SCHEDULER)
    //     .set("", "kvcache_manager", kvcache_manager_.get());
  }
  return true;
}

bool Batching::contiguous_batch(dicts& input_data, const size_t input_data_size,
                                const size_t new_pop, dicts& redundant_data) {
  if (!request_states_->wait_all_ready(200)) {
    // SPDLOG_INFO("contiguous_batching: not all requests ready. Batch sz={}, Req sz = {}",
    //             input_data_size + new_pop, request_states_->size());
    return false;
  } else {
    auto qs = input_queue_.size();
    SPDLOG_INFO(
        "contiguous_batching: check. Batch sz={}, Req sz = {}, input_data sz={}, "
        "max_batch_size_={}, qs={}",
        input_data_size + new_pop, request_states_->size(), input_data.size(), max_batch_size_, qs);
    if (qs != 0 && (input_data_size + new_pop < max_batch_size_)) return false;  // rebatching
  }

  SPDLOG_INFO("contiguous_batching: all requests ready. Req sz={}, state sz = {}",
              input_data_size + new_pop, request_states_->size());

  TimeGuard guard("contiguous_batching");
  // for (const auto& request : input_data) {
  //   auto iter = request->find("request_id");
  //   if (iter == request->end()) {
  //     SPDLOG_ERROR("request_id not found in contiguous batching mode");
  //     return false;
  //   }

  //   std::string* request_id = any_cast<std::string>(&iter->second);

  //   if (request_states_->add_iter_index(*request_id) == 0) {
  //   }
  // }
  for (auto iter = input_data.begin(); iter != input_data.end(); ++iter) {
    auto iter_id = (*iter)->find("request_id");

    std::string* request_id = any_cast<std::string>(&iter_id->second);
    // if (request_states_->add_iter_index(*request_id) == 0) {
    //   kvcache::KVCacheAllocParams request_params;
    //   auto& storage = ThreadSafeKVStorage::getInstance().get(*request_id);

    //   const llm::SamplingParams* samp =
    //       any_cast<llm::SamplingParams>(&storage.get("sampling_params"));

    //   request_params.max_new_tokens = samp->max_tokens;
    //   request_params.request_id = *request_id;
    //   request_params.kvcache_seq_len = request_states_->get_kvcache_seq_len(*request_id);

    //   // request_params.max_new_tokens;
    //   SPDLOG_WARN("contiguous_batching: alloc_reqid: request_id={}, kvcache_seq_len={}",
    //               request_params.request_id, request_params.kvcache_seq_len);
    //   // kvcache_manager_->alloc_reqid(request_params);
    //   // request_states_.emplace(request_id, RequestState({true, request_size}));
    //   // num_prefill++;
    // }
  }
  // auto valid_reqs = kvcache_manager_->step();
  // IPIPE_ASSERT(!valid_reqs.empty());

  for (auto iter = input_data.begin(); iter != input_data.end();) {
    auto iter_id = (*iter)->find("request_id");

    std::string* request_id = any_cast<std::string>(&iter_id->second);
    // SPDLOG_INFO("check req: request_id={}", *request_id);
    if (false) {
      redundant_data.push_back(*iter);
      iter = input_data.erase(iter);
    } else {
      ++iter;
      // SPDLOG_INFO("contiguous_batching: request_id={} is valid", *request_id);
      request_states_->set_unready(*request_id);
    }
  }
  // SPDLOG_INFO("contiguous_batching: valid_reqs sz={}, input_data sz={}, redundant_data sz={}",
  //             valid_reqs.size(), input_data.size(), redundant_data.size());
  // IPIPE_ASSERT(valid_reqs.size() == input_data.size());  // 必须保证一致

  return true;
}

void Batching::run() {  // only one Batching thread

  std::vector<dict> input_data;
  std::vector<dict> redundant_data;
  while (bThreadInited_.load()) {
    {
      // TimeGuard guard;
      if (!batched_queue_.WaitForWaiting(100)) continue;

      // auto time_pass = guard.elapsed();
      // if (time_pass > 16) {
      //   SPDLOG_WARN("WaitForWaiting too slow: {}", time_pass);
      // }
      // guard.silence();
    }
    std::vector<dict> redundant_data;

    auto input_data_size = get_request_size(input_data);

    const auto data_size = input_queue_.size();

    if (data_size + input_data_size >= max_batch_size_) {
      // for (uint32_t i = 0; i < max_batch_size_ - input_data.size(); ++i) {
      //   input_data.push_back(input_queue_.WaitPop());
      // }
      std::size_t new_pop = 0;
      while (input_data_size + new_pop < max_batch_size_) {
        new_pop += input_queue_.front_size();
        if (input_data_size + new_pop > max_batch_size_) {
          break;
        }
        input_data.push_back(input_queue_.WaitPop());
      }

      if (request_states_) {
        if (!contiguous_batch(input_data, input_data_size, new_pop, redundant_data)) continue;
      }

      backend_->forward(input_data);
    } else if (data_size + input_data.size() == 0) {
      dict tmp_dict;
      if (!input_queue_.WaitForPop(tmp_dict,
                                   batching_timeout_)) {  // every batching_timeout_ ms check that
                                                          // whether bIbited_ is true.
        // if not, exit this  loop and thread
        continue;
      }
      input_data.push_back(tmp_dict);
      continue;

    } else {
      std::size_t new_pop = 0;
      // 保证input_data里有至少一个
      if (input_data.empty()) {
        IPIPE_ASSERT(!input_queue_.empty());
        new_pop += input_queue_.front_size();
        input_data.push_back(input_queue_.WaitPop());
        // it is guaranteed that new_pop < max_batch_size_ for one input
      }
      std::shared_ptr<SimpleEvents> event =
          any_cast<std::shared_ptr<SimpleEvents>>(input_data[0]->at(TASK_EVENT_KEY));
      auto time_es = event->time_passed();

      if (time_es < batching_timeout_ && !runing_state_->skip_waiting_for_batching()) {
        input_queue_.Wait(int(batching_timeout_ - time_es));
        // #ifndef NDEBUG
        //         auto time_es2 = event->time_passed();
        //         SPDLOG_DEBUG("time for batching:{}", time_es2);
        // #endif
        continue;
      }

      while (!input_queue_.empty() && (input_data_size + new_pop < max_batch_size_)) {
        new_pop += input_queue_.front_size();
        if (input_data_size + new_pop > max_batch_size_) {
          // new_pop -= input_queue_.front_size();
          break;
        }
        input_data.push_back(input_queue_.WaitPop());
      }
      if (input_data_size + new_pop >= max_batch_size_) {
        continue;  // go to another branch
      }

      if (request_states_) {
        if (!contiguous_batch(input_data, input_data_size, new_pop, redundant_data)) continue;
      }

      backend_->forward(input_data);
    }
    input_data.clear();
    input_data.swap(redundant_data);
  }  // end while
};
IPIPE_REGISTER(Backend, Batching, "Batching");

}  // namespace ipipe
