// Copyright 2021-2025 NetEase.
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

#include <mutex>

#include "omniback/core/event.hpp"
#include "omniback/helper/timer.hpp"

namespace omniback {

Event::Event(size_t num) : num_task(num), starttime_(helper::now()) {}
float Event::time_passed() {
  return helper::time_passed(starttime_);
}

} // namespace omniback
