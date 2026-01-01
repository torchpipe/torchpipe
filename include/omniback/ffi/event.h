#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

namespace omniback::ffi {

namespace tf = tvm::ffi;
namespace refl = tf::reflection;

class EventObj : public tf::Object {
 public:
  EventObj(uint32_t num = 1);

  void wait_danger() {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { return (num_task == ref_count) || eptr_; });
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
  }

  uint32_t reset(uint32_t num) {
    {
      std::unique_lock<std::mutex> lk(mut);
      num_task = num;
      std::swap(num_task, num);
    }

    try_callback();

    data_cond.notify_all();
    return num;
  }

  bool wait_finish(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mut);

    return data_cond.wait_for(
        lk, std::chrono::milliseconds(timeout_ms), [this] {
          return (num_task == ref_count);
        }); //
  }

  /// true: 引用计数等小于引用计数目标值时。
  bool valid() {
    std::lock_guard<std::mutex> lk(mut);
    return num_task > ref_count;
  }

  /// @brief 阻塞，以条件引用计数等于引用计数目标值进行监听。
  /// @return std::exception_ptr 异常
  std::exception_ptr wait_and_get_except() {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { return (num_task == ref_count); }); //
    return eptr_;
  }

  void wait() {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { return (num_task == ref_count); }); //
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
  }

  void try_throw() {
    std::unique_lock<std::mutex> lk(mut);

    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
  }

  void wait_finish() {
    std::unique_lock<std::mutex> lk(mut);

    data_cond.wait(lk, [this] { return (num_task == ref_count); }); //
  }

  bool wait(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mut);

    bool done =
        data_cond.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
          return (num_task == ref_count);
        }); //
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
    return done;
  }

  bool wait_danger(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mut);

    bool done =
        data_cond.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this] {
          return (num_task == ref_count) || eptr_;
        }); //
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
    return done;
  }

  void notify_one() {
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add();
    }
    try_callback();

    data_cond.notify_one();
  }

  void notify_one(uint32_t num) {
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add(num);
    }
    try_callback();

    data_cond.notify_one();
  }
  /**
   * @brief 通知所有监听者。
   * @ref EventObj::notify_one
   */
  void notify_all() {
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add();
    }
    try_callback();

    data_cond.notify_all();
  }

  void set() {
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add();
    }
    try_callback();

    data_cond.notify_all();
  }

  bool is_set() {
    std::lock_guard<std::mutex> lk(mut);
    return ref_count == num_task;
  }

  void notify_all_after_check_exception(EventObj * extra_ev) {
    if (extra_ev->has_exception())
      this->set_exception_and_notify_all(extra_ev->reset_exception());
    else
      this->notify_all();
  }

  void notify_all(uint32_t num) {
    {
      // do not remove this lock even if ref_count is atomic
      std::lock_guard<std::mutex> lk(mut);
      ref_add(num);
    }
    try_callback();

    data_cond.notify_all();
  }

  /// @brief  是否被设置了异常。
  bool has_exception() {
    std::lock_guard<std::mutex> lk(mut);
    return eptr_ != nullptr;
  }

  void set_exception_and_notify_one(std::exception_ptr eptr) {
    {
      std::lock_guard<std::mutex> lk(mut);

      if (!eptr_)
        eptr_ = eptr;
      ref_add();
    }
    try_callback();

    data_cond.notify_one();
  }
  void ref_add() {
    if (ref_count < num_task)
      ref_count++;
    else {
      assert(false);
    }
  }

  void ref_add(uint32_t num) {
    if (ref_count < num_task)
      ref_count += num;
    else {
      assert(false);
    }
    assert(ref_count <= num_task);
  }

  /**
   * @brief 清理并返回被设置的异常
   */
  std::exception_ptr reset_exception() {
    std::lock_guard<std::mutex> lk(mut);
    auto re = eptr_;
    eptr_ = nullptr;
    return re;
  }

  std::exception_ptr get_exception() {
    std::lock_guard<std::mutex> lk(mut);

    return eptr_;
  }

  void task_add(uint32_t num) {
    std::lock_guard<std::mutex> lk(mut);
    num_task += num;
  }
  void try_callback() {
    bool should_try = false;
    std::vector<std::function<void(std::exception_ptr)>> excep_cb;
    {
      std::lock_guard<std::mutex> lk(mut);
      should_try = (ref_count >= num_task);
      std::swap(excep_cb, exception_callbacks_);
    }

    if (should_try) {
      if (eptr_ && !excep_cb.empty()) { // no need to lock the eptr_ now
        while (!excep_cb.empty()) {
          excep_cb.back()(eptr_); // Execute the last callback
          excep_cb.pop_back(); // Remove the last callback
        }
        eptr_ = nullptr;
      }
      while (!callbacks_.empty()) {
        callbacks_.back()(); // Execute the last callback
        callbacks_.pop_back(); // Remove the last callback
      }
      while (!latest_callbacks_.empty()) {
        latest_callbacks_.back()(); // Execute the last callback
        latest_callbacks_.pop_back(); // Remove the last callback
      }
    }
  }

  /// 参见 @ref EventObj::set_exception_and_notify_one
  void set_exception_and_notify_all(std::exception_ptr eptr) {
    {
      std::lock_guard<std::mutex> lk(mut);

      if (!eptr_)
        eptr_ = eptr;
      ref_add();
    }
    try_callback();

    data_cond.notify_all();
  }

  /// 设置回调函数
  void append_callback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lk(mut);
    // assert(!callback_);
    callbacks_.emplace_back(callback);
  }

  void set_callback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lk(mut);
    // assert(!callback_);
    {
      callbacks_.push_back(callback);
    }
  }

  void set_exception_callback(
      const std::function<void(std::exception_ptr)>& callback) {
    std::lock_guard<std::mutex> lk(mut);
    // assert(!callback_);
    {
      exception_callbacks_.push_back(callback);
    }
  }

  void set_final_callback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lk(mut);
    // assert(!callback_);

    latest_callbacks_.push_back(callback);
    if (latest_callbacks_.size() > 1) {
      latest_callbacks_.pop_back();
      throw std::runtime_error(
          "The callback stack is not empty. `set_final_callback` is used "
          "to set the latest unique "
          "callback. Consider using set_callback instead.");
    }
  }
  void clear_callback() {
    latest_callbacks_.clear();
    callbacks_.clear();
  }

  /// 获得从构造到现在经过的时间（单位：毫秒）。
  float time_passed();

  ~EventObj() {
  };

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("omniback.Event", EventObj, tf::Object);

 private:
  std::mutex mut; // mutable
  std::condition_variable data_cond;
  uint32_t ref_count = 0;
  uint32_t num_task;
  std::vector<std::function<void(std::exception_ptr)>> exception_callbacks_;
  std::vector<std::function<void()>> callbacks_;
  std::vector<std::function<void()>> latest_callbacks_;

  std::exception_ptr eptr_;
  std::chrono::steady_clock::time_point starttime_;
};

class Event : public tf::ObjectRef {
 public:
  explicit Event(uint32_t num = 1) {
    data_ = tf::make_object<EventObj>(num);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(
      Event,
      tf::ObjectRef,
      EventObj);
};



} // namespace omniback