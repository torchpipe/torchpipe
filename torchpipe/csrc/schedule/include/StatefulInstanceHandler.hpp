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

#include <atomic>
#include <cassert>
#include <memory>
#include <numeric>
#include <string>

#include <vector>
#include "time_utils.hpp"

#include <algorithm>
#include "Backend.hpp"
#include "dict.hpp"
#include "params.hpp"
#include "reflect.h"
// #include "state_event.hpp"
#include "threadsafe_queue_sized.hpp"
namespace ipipe {

/**
 * @brief 后端的执行线程。
 *
 *
 *
 */
class StatefulInstanceHandler : public Backend {
 public:
  ~StatefulInstanceHandler() {
    bInited_.store(false);

    if (thread_.joinable()) thread_.join();
  }

  /**
   * @brief 初始化，并启动执行线程 StatefulInstanceHandler::run.
   *
   * @param backend 真正的运算后端
   * @param _independent_thread_index 实例编号。
   * @param _batched_queue ThreadSafeSizedQueue<std::vector<dict>>*
   * 数据队列，可以从中获取数据。
   *
   * @param _state_event std::shared_ptr<StateEvents> 向其反馈状态。
   */
  bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
    params_ =
        Params(std::unordered_map<std::string, std::string>{},  // 初始化参数：含默认值
               std::set<std::string>{
                   "backend", "_independent_thread_index"},  // 初始化参数：不含默认值， 必须提供
               std::unordered_map<std::string, std::string>{},  // forward参数：不含默认值，
                                                                // 必须提供
               std::set<std::string>{});  // forward参数：不含默认值， 必须提供
    if (!params_.init(config)) return false;
    batched_queue_ =
        any_cast<ThreadSafeSizedQueue<std::vector<dict>>*>(dict_config->at("_batched_queue"));
    independent_thread_index_ = std::stoi(params_.at("_independent_thread_index"));

    config_ = &config;
    dict_config_ = dict_config;

    // auto iter = dict_config->find("_state_event");
    // if (iter != dict_config->end()) {
    //   state_ = any_cast<std::shared_ptr<StateEvents>>(iter->second);
    // }

    thread_ = std::thread(&StatefulInstanceHandler::run, this);
    while (!bInited_.load() && (!bStoped_.load())) {
      std::this_thread::yield();
    }
    if (init_eptr_) std::rethrow_exception(init_eptr_);
    return (bInited_.load() && (!bStoped_.load()));
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  void forward(const std::vector<dict>& input_dicts) override final {
    // IPIPE_ASSERT(False);
    assert(false);
    throw std::runtime_error("not accessable");
    // const std::size_t size = get_request_size(input_dicts);
    // batched_queue_->Push(input_dicts, size);
  }
  virtual uint32_t max() const {
    return engine_->max();
  };  // 用于判断最大形状 // get_shapes_restrict 的简单情形
  virtual uint32_t min() const { return engine_->min(); };
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
 private:
#endif

  /**
   * @brief 将有单独线程执行此函数。
   * 计算后端的初始化和前行都将在此线程中执行。
   *
   * 如果编译系统定义了宏
   * NCATCH_SUB，将不会转发计算后端初始化和前向过程中的异常。（用于测试）
   *
   * 如果数据中存在 TASK_EVENT_KEY， 将通过事件传递异常或者结果。
   *
   */
  virtual void run();

 protected:
  Params params_;
  std::thread thread_;
  ThreadSafeSizedQueue<std::vector<dict>>* batched_queue_ = nullptr;

  std::unique_ptr<Backend> engine_;  // 资源所有权
  std::atomic_bool bInited_{false};
  std::atomic_bool bStoped_{false};
  std::atomic_bool bBusy_{false};
  const std::unordered_map<std::string, std::string>* config_;
  dict dict_config_;

  std::exception_ptr init_eptr_;
  // std::shared_ptr<StateEvents> state_;
  int independent_thread_index_ = 0;
};

}  // namespace ipipe