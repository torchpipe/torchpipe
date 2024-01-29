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

#include "Backend.hpp"
#include "dict.hpp"

#include <memory>
#include "Sequential.hpp"

namespace ipipe {
class Params;

/**
 * @brief 此后端可以管理采用了<a
 * href="https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams">pytorch的cuda流管理方式</a>
 * 的后端的流的同步时机：当所有此类计算后端经过调度后在背景线程中执行时，除非本身已经实现了相应同步功能，可以借助此设施在Sequential等容器中完成流同步。
 *
 * @note
 * 调度程序初始化该后端时，需指定_independent_thread_index，以便 TensorSync 后端生效。
 *

 */
class TensorSync : public Backend {
 public:
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  virtual void forward(const std::vector<dict>&) override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
  virtual uint32_t max() const { return engine_->max(); }

  virtual uint32_t min() const { return engine_->min(); }
#endif

 private:
  std::unique_ptr<Params> params_;

  std::unique_ptr<Backend> engine_;
  bool bNeedSync_ = false;
};
}  // namespace ipipe
