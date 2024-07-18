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

#include "StatefulInstanceHandler.hpp"
#include <condition_variable>
#include <numeric>
#include <sstream>

#include "base_logging.hpp"
#include "event.hpp"
#include "reflect.h"
#include "time_utils.hpp"
#include "exception.hpp"

namespace ipipe {

void StatefulInstanceHandler::run() {  // 确保初始化和前向处于同一个线程中
  std::string backend_name = params_.at("backend");

#ifndef NCATCH_SUB
  try {
#endif
    IPIPE_ASSERT(config_);
    engine_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, backend_name));
    auto iter_backend_back = config_->find(backend_name + "::backend");
    if (iter_backend_back != config_->end()) {
      backend_name = backend_name + '[' + iter_backend_back->second + ']';
    }
    if (engine_ && engine_->init(*config_, dict_config_)) {
      bInited_.store(true);
    } else {
      bInited_.store(false);
      engine_.reset();
    }

#ifndef NCATCH_SUB
  } catch (const std::exception& e) {
    bInited_.store(false);
    engine_.reset();
    SPDLOG_ERROR("Backend initialization: {}", e.what());
    init_eptr_ = std::current_exception();
  }
#endif
  const auto max_len = engine_ ? engine_->max() : 0;
  const auto min_len = engine_ ? engine_->min() : 0;
  while (bInited_.load()) {
    std::vector<dict> tasks;
    {
      auto succ = batched_queue_->WaitForPopWithSize(tasks, 50, [max_len, min_len](std::size_t in) {
        return in >= min_len && in <= max_len;
      });  // for exit this thread
      if (!succ) {
        assert(tasks.empty());
        continue;
      }
    }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      tasks[i]->erase(TASK_RESULT_KEY);
    }
    // StateEventsGuard state_guard(state_ ? state_.get() : nullptr);
    std::vector<std::shared_ptr<SimpleEvents>> events;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto iter_time = tasks[i]->find(TASK_EVENT_KEY);
      if (iter_time != tasks[i]->end()) {
        std::shared_ptr<SimpleEvents> ti_p =
            any_cast<std::shared_ptr<SimpleEvents>>(iter_time->second);
        events.push_back(ti_p);
      } else {
        std::string node_name;
        auto iter_name = tasks[i]->find("node_name");
        if (iter_name != tasks[i]->end()) {
          node_name = any_cast<std::string>(iter_name->second);
        }
        SPDLOG_ERROR(node_name + ": Fatal Error: lack of event.");
      }
    }
    if (events.size() != tasks.size()) {
      for (auto& item : events) {
        std::runtime_error error("lack of event.");
        item->set_exception_and_notify_all(make_exception_ptr(error));
      }
      continue;
    }
#ifndef NCATCH_SUB
    try {
#endif

      if (tasks.size() < min() || tasks.size() > max()) {
        std::stringstream ss;
        ss << "tasks.size() < min() || tasks.size() > max(): " << "tasks.size() = " << tasks.size()
           << " min=" << min() << " max=" << max();
        IPIPE_THROW(ss.str());
      }
      for (auto item : tasks) {
        item->erase(TASK_EVENT_KEY);
      }
      engine_->forward(tasks);

#ifndef NCATCH_SUB
    } catch (const std::exception& e) {
      //  notify callback
      std::exception_ptr eptr =
          insert_exception(e.what(), " while processing backend `" + backend_name +
                                         '`');  // std::current_exception();  // 捕获
      for (std::size_t i = 0; i < tasks.size(); ++i) {
        (*tasks[i])[TASK_EVENT_KEY] = events[i];
      }
      for (std::size_t i = 0; i < tasks.size(); ++i) {
        events[i]->set_exception_and_notify_all(eptr);
      }
      continue;
    }
#endif
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      (*tasks[i])[TASK_EVENT_KEY] = events[i];
    }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      events[i]->notify_all();
    }
  }
  bStoped_.store(true);
}

}  // namespace ipipe