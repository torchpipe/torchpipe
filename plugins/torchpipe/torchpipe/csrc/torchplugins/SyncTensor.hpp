#pragma once

#include <unordered_set>
#include <string>
#include <hami/extension.hpp>

#include "c10/cuda/CUDAGuard.h"

using hami::dict;

namespace torchpipe {
class SyncTensor : public hami::DependencyV0 {
 public:
  /**
   * @brief
   * Initialization, determines whether the default stream is being used; if
   * so, and in independent thread mode, it will bind a new CUDA stream to the
   * current thread for GPU asynchronous execution. (Renamed from Torch)
   *
   * @param TASK_INDEX_KEY When the parameter is not null, it represents
   * independent thread mode, at this time it can be assumed that init and
   * forward are running in the same independent thread. Check if the CUDA
   * stream is the default stream, if not, we bind the thread to a new stream
   * and set bNeedSync_ to true, otherwise do nothing.
   * @param SyncTensor::backend Default is Identity. The backend it forwards
   * execution to.
   * @note Usage: SyncTensor[A], SequentialV0[A,B,C,SyncTensor] or
   * SequentialV0[A,B,SyncTensor[C]]. For serial units, such as
   * SequentialV0[SyncTensor[A],SyncTensor[B]], it will initialize in reverse
   * order and forward in order: SyncTensor[B].init -> SyncTensor[A].init ->
   * SyncTensor[A].forward
   * -> SyncTensor[B].forward;  SyncTensor[A] is not the default stream at
   * initialization, so it does not need to set a new stream, and it does not
   * need to be responsible for stream synchronization during forward, at this
   * time if SyncTensor[B] has set a new stream, then SyncTensor[B] is
   * responsible for stream synchronization;
   *
   *  @ref SequentialV0 and other containers can ensure that the
   * initialization order of their child backends is opposite to the forward
   * order, therefore, even in complex situations, the correct stream
   * synchronization timing can be obtained.
   */
  virtual void pre_init(
      const std::unordered_map<std::string, std::string>&,
      const dict&) override;

  virtual void post_init(
      const std::unordered_map<std::string, std::string>&,
      const dict&) override;
  /**
   * @brief
   * 如果init时绑定了新的流（也就是bNeedSync_==true），则forward时当子后端执行完毕后执行当前流上的同步。
   *
   */
  virtual void custom_forward_with_dep(const std::vector<dict>&, Backend*)
      override;

 private:
  bool bNeedSync_ = false;
};

class StreamGuard : public hami::DependencyV0 {
 public:
  /**
   * @brief
   * Initialization, determines whether the default stream is being used; if
   * so, and in independent thread mode, it will bind a new CUDA stream to the
   * current thread for GPU asynchronous execution. (Renamed from Torch)
   *
   * @param TASK_INDEX_KEY When the parameter is not null, it represents
   * independent thread mode, at this time it can be assumed that init and
   * forward are running in the same independent thread. Check if the CUDA
   * stream is the default stream, if not, we bind the thread to a new stream
   * and set bNeedSync_ to true, otherwise do nothing.
   * @param StreamGuard::backend Default is Identity. The backend it forwards
   * execution to.
   *
   *  @ref SequentialV0 and other containers can ensure that the
   * initialization order of their child backends is opposite to the forward
   * order, therefore, even in complex situations, the correct stream
   * synchronization timing can be obtained.
   */
  virtual void pre_init(
      const std::unordered_map<std::string, std::string>&,
      const dict&) override;

  virtual void post_init(
      const std::unordered_map<std::string, std::string>&,
      const dict&) override;
  /**
   * @brief
   * 如果init时绑定了新的流（也就是bNeedSync_==true），则forward时当子后端执行完毕后执行当前流上的同步。
   *
   */
  virtual void custom_forward_with_dep(const std::vector<dict>&, Backend*)
      override;

 private:
  //   bool bNeedSync_ = false;
  std::optional<c10::cuda::CUDAStream> stream_;
  std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard_;
};
} // namespace torchpipe