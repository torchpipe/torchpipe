#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace omniback {

class InitializationRegistry {
 public:
  static InitializationRegistry& instance();

  // Register an initialization function with a name
  void register_initialization(
      const std::string& name,
      std::function<void()> init_func);

  // Try to execute all pending initializations
  void try_initialization();

  // Prevent copying and moving
  InitializationRegistry(const InitializationRegistry&) = delete;
  InitializationRegistry& operator=(const InitializationRegistry&) = delete;
  InitializationRegistry(InitializationRegistry&&) = delete;
  InitializationRegistry& operator=(InitializationRegistry&&) = delete;

 protected:
  InitializationRegistry() = default;

 private:
  struct InitFunction {
    std::function<void()> func;
    bool executed{false};
  };

  std::unordered_map<std::string, InitFunction> init_functions_;
  mutable std::mutex mutex_;
};

// Helper function to register initialization
void register_initialization(
    const std::string& name,
    std::function<void()> init_func);

// Helper function to try initialization
void try_initialization();

} // namespace omniback