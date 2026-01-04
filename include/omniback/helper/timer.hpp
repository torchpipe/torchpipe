

#pragma once
#include <chrono>
#include <string>

#define SHUTDOWN_TIMEOUT 500
#define SHUTDOWN_TIMEOUT_MS std::chrono::milliseconds(SHUTDOWN_TIMEOUT)

namespace omniback::helper {

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

inline std::chrono::steady_clock::time_point now() {
  return std::chrono::steady_clock::now();
}

std::chrono::steady_clock::time_point start_time();

inline float timestamp() {
  std::chrono::duration<float, std::milli> fp_ms = now() - start_time();
  return fp_ms.count();
}

inline float time_passed(decltype(now()) time_old) {
  std::chrono::duration<float, std::milli> fp_ms = now() - time_old;
  return fp_ms.count();
}
} // namespace omniback::helper