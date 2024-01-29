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
 * 调度程序初始化该后端时，需指定_independent_thread_index，以便 SyncTensor 后端生效。
 *

 */
class SyncTensor : public Backend {
 public:
  /**
   * @brief
   * Initialization, determines whether the default stream is being used; if so, and in independent
   * thread mode, it will bind a new CUDA stream to the current thread for GPU asynchronous
   * execution. (Renamed from Torch)
   *
   * @param _independent_thread_index When the parameter is not null, it represents independent
   * thread mode, at this time it can be assumed that init and forward are running in the same
   * independent thread. Check if the CUDA stream is the default stream, if not, we bind the thread
   * to a new stream and set bNeedSync_ to true, otherwise do nothing.
   * @param SyncTensor::backend Default is Identity. The backend it forwards execution to.
   * @note Usage: SyncTensor[A], Sequential[A,B,C,SyncTensor] or Sequential[A,B,SyncTensor[C]].
   * For serial units, such as Sequential[SyncTensor[A],SyncTensor[B]],
   * it will initialize in reverse order and forward in order: SyncTensor[B].init ->
   * SyncTensor[A].init -> SyncTensor[A].forward
   * -> SyncTensor[B].forward;  SyncTensor[A] is not the default stream at initialization,
   * so it does not need to set a new stream, and it does not need to be responsible for stream
   * synchronization during forward, at this time if SyncTensor[B] has set a new stream, then
   * SyncTensor[B] is responsible for stream synchronization;
   *
   *  @ref Sequential and other containers can ensure that the initialization order of their child
   * backends is opposite to the forward order, therefore, even in complex situations, the correct
   * stream synchronization timing can be obtained.
   */
  virtual bool init(const std::unordered_map<std::string, std::string>&, dict) override;

  /**
   * @brief
   * 如果init时绑定了新的流（也就是bNeedSync_==true），则forward时当子后端执行完毕后执行当前流上的同步。
   *
   */
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
