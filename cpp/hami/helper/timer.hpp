

#pragma once
#include <chrono>
#include <string>

#define SHUTDOWN_TIMEOUT 500
#define SHUTDOWN_TIMEOUT_MS std::chrono::milliseconds(SHUTDOWN_TIMEOUT)

namespace hami::helper {

class ScopedTimer {
 public:
  explicit ScopedTimer(const std::string& name, float probability = 1.0)
      : name_(name),
        probability_(probability),
        start_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer();

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
  ScopedTimer(ScopedTimer&&) = delete;
  ScopedTimer& operator=(ScopedTimer&&) = delete;

 private:
  std::string name_;
  float probability_{1};
  std::chrono::steady_clock::time_point start_;
};
}  // namespace hami::helper