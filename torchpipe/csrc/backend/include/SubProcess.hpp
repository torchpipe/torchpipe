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

#include <memory>

namespace ipipe {
class Params;

/**
 * @brief 此后端可以管理采用了<a
 * href="https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams">pytorch的cuda流管理方式</a>
 * 的后端的流的同步时机：当所有此类计算后端经过调度后在背景线程中执行时，除非本身已经实现了相应同步功能，可以借助此设施在Sequential等容器中完成流同步。
 *
 * @note
 * 调度程序初始化该后端时，需指定_independent_thread_index，以便 SubProcess 后端生效。
 *

 */
class SubProcess : public Backend {
 public:
  /**
   * @brief
   * 初始化，判断是否正在使用默认流；如果是，而且在独立线程模式时，将绑定新的cuda流到当前线程，用于gpu异步执行。(从
   * Torch 更名而来)
   *
   * @param _independent_thread_index 参数非空时，代表独立线程模式，
   * 此时可认为init和forward位于同一个独立的线程中运行。检查cuda流是否是默认流，如果不是，我们绑定线程到新的流，并置bNeedSync_为true，
   * 否则什么都不做。
   * @param SubProcess::backend 默认 Identity. 他所转发执行的后端。
   * @note 用法： SubProcess[A], Sequential[A,B,C,SubProcess] 或者
   * Sequential[A,B,SubProcess[C]].
   * 对于串行单元， 比如 Sequential[SubProcess[A],SubProcess[B]],
   * 会倒序初始化, 正序前向： SubProcess[B].init -> SubProcess[A].init -> SubProcess[A].forward
   * -> SubProcess[B].forward；  SubProcess[A] 在
   * 初始化时已经不是默认流，则它不用设置新的流， forward时也不用负责流的同步,
   * 此时如果 SubProcess[B] 设置了新的流，则由 SubProcess[B] 负责流的同步;
   *
   *  @ref Sequential 等容器可确保其子后端初始化的顺序和前向的顺序相反，
   * 因此即使在复合情况下，也能获得正确的流同步时机。
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
