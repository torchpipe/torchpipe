#pragma once

#include <stdexcept>

namespace omniback::error {

class ExceptionHolder {
 public:
  explicit ExceptionHolder(std::exception_ptr ptr) : ptr_(std::move(ptr)) {}

  bool has_exception() const noexcept {
    return static_cast<bool>(ptr_);
  }

  void rethrow() const {
    if (!has_exception()) {
      throw std::runtime_error("No exception stored");
    }
    std::rethrow_exception(ptr_);
  }
  // std::exception_ptr get_exception_ptr() const { return ptr_; }
  operator std::exception_ptr() {
    return ptr_;
  }

 private:
  std::exception_ptr ptr_;
};

class KeyNotFoundError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class NoResultError : public KeyNotFoundError {
 public:
  NoResultError() : KeyNotFoundError("result is empty") {}
  using KeyNotFoundError::KeyNotFoundError;
};

} // namespace omniback::error