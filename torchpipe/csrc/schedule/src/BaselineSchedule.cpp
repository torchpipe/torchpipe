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

#include "BaselineSchedule.hpp"

#include <chrono>
#include <functional>
#include <sstream>
#include <thread>
#include "base_logging.hpp"
#include "Backend.hpp"
#include "dict.hpp"
#include "Instances.hpp"
#include "time_utils.hpp"
namespace ipipe {

BaselineSchedule::~BaselineSchedule() {
  bThreadInited_.store(false);
  if (!input_queue_.empty()) {
    SPDLOG_ERROR("!input_queue_.empty()");
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void BaselineSchedule::run() {  // only one BaselineSchedule thread

  std::vector<dict> input_data;

  while (bThreadInited_.load()) {
    auto data_size = input_queue_.size();

    if (data_size + input_data.size() >= max_batch_size_) {
      // for (uint32_t i = 0; i < max_batch_size_ - input_data.size(); ++i) {
      //   input_data.push_back(input_queue_.WaitPop());
      // }
      while (input_data.size() < max_batch_size_) {
        input_data.push_back(input_queue_.WaitPop());
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
      // 保证input_data里有至少一个
      if (!input_queue_.empty()) {
        input_data.push_back(input_queue_.WaitPop());
      }
      std::shared_ptr<SimpleEvents> event =
          any_cast<std::shared_ptr<SimpleEvents>>(input_data[0]->at(TASK_EVENT_KEY));
      auto time_es = event->time_passed();

      if (time_es < batching_timeout_ && !runing_state_->skip_waiting_for_batching()) {
        input_queue_.Wait(int(batching_timeout_ - time_es));
        continue;
      }

      while (!input_queue_.empty() && (input_data.size() < max_batch_size_)) {
        input_data.push_back(input_queue_.WaitPop());
      }
      backend_->forward(input_data);
    }
    input_data.clear();
  }  // end while
};
IPIPE_REGISTER(Backend, BaselineSchedule, "BaselineSchedule");

}  // namespace ipipe
