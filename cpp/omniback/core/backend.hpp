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

#pragma once
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "omniback/core/dict.hpp"
#include "omniback/core/reflect.h"
#include "omniback/helper/omniback_export.h"

namespace omniback {

/**
 * @brief Base class for all backends.
 *
 * Provides the interface for initializing and processing data.
 *  todo: Non-Virtual Interface (NVI)
 */
class OMNI_EXPORT Backend {
 public:
  // Non-Virtual Interface
  /**
   * @brief Initializes the backend with configuration and shared data.
   *
   * Must be called before any other operations. After successful
   * initialization, the values of `max()` and `min()` will not change.
   *
   * @param params Configuration parameters for the backend.
   * @param options Shared data accessible across multiple backends.
   * @throws any Exception thrown during initialization.
   * @note Non-Virtual Interface. Use can implement @impl_init
   */
  void init(
      const std::unordered_map<string, string>& params,
      const dict& options) {
    impl_init(params, options);
  }

  /**
   * @brief Processes input/output data.
   *
   * [min(), max()] give a hit for The range of `io`.
   *
   * @param io Input/output data to be processed.
   */
  void forward(const std::vector<dict>& io) {
    impl_forward(io);
  }

  /**
   * @brief Returns the maximum number of inputs supported.
   *
   * This value is determined after `init()` and `inject_dependency` are
   * called.
   *
   * @return Maximum number of inputs supported. Default to
   * std::numeric_limits<size_t>::max()
   */
  [[nodiscard]] size_t max() const {
    return impl_max();
  }

  /**
   * @brief Returns the minimum number of inputs supported.
   *
   * This value is determined after `init()` and `inject_dependency` are
   * called.
   *
   * @return Minimum number of inputs supported.
   */
  [[nodiscard]] size_t min() const {
    return impl_min();
  }

  /**
   * @brief Sets the dependency for the backend.
   *
   * This method allows setting a Backend object as a dependency for the
   * current backend. It should be called before the normal forward method if
   * a dependency is required. Backend implementers can use this dependency to
   * perform actual work. This is different from the dependency in the forward
   * method, which is a runtime dependency.
   *
   * This is primarily used for Inversion of Control (IoCV0) or Dependency
   * Injection (DI_v0) in the scheduling system. For typical computation-type
   * backends, this can be ignored.
   *
   * @param dependency Pointer to the backend dependency.
   * @throws std::runtime_error If dependency setting is not supported by this
   * backend.
   */
  void inject_dependency(Backend* dependency) {
    impl_inject_dependency(dependency);
  }

  /**
   * @brief Processes input/output data with a runtime dependency.
   *
   * @param io Input/output data to be processed.
   * @param dependency Pointer to the backend dependency.
   */
  void forward_with_dep(const std::vector<dict>& io, Backend& dependency) {
    impl_forward_with_dep(io, dependency);
  }

 public:
  // helper function
  virtual ~Backend() = default;
  Backend() = default;
  Backend(const Backend&) = delete;
  Backend(const Backend&&) = delete;
  Backend& operator=(const Backend&) = delete;
  Backend& operator=(const Backend&&) = delete;

  /**
   * @brief Helper function for @ref forward to check the range of [min(),
   * max()].
   *
   * This function ensures that the size of `io` is within the valid
   * range defined by min() and max(). If the size is within the range, it
   * calls the forward method. If the size exceeds max(), it processes the
   * input in chunks. If the size is less than min(), it throws an
   * invalid_argument exception.
   *
   * @param io The input/output data to be processed.
   * @throws std::invalid_argument if the size of io is less than
   * min().
   */
  void safe_forward(const std::vector<dict>& io);

 private:
  /**
   * @brief Optional helper function for reflection.
   *
   * This function is useful for obtaining the class name of backend objects
   * that are not generated through reflection and has multiple class names(like
   * Sequential && S). Backends created using OMNI_CREATE do not require this
   * function.
   *
   * @return The default class name of the backend.
   */
  virtual std::string reflect_cls_name() const {
    return "UnReflectClassIDontKnowTheClassName";
  }

 private:
  // User API
  virtual void impl_init(
      const std::unordered_map<string, string>& params,
      const dict& options) {}
  virtual void impl_forward(const std::vector<dict>& ios) {
    impl_forward_with_dep(ios, *this);
  }

  virtual void impl_forward_with_dep(
      const std::vector<dict>& ios,
      Backend& dependency) {
    throw std::runtime_error(
        "Neither impl_forward nor impl_forward_with_dep is implemented");
  }

  [[nodiscard]] virtual size_t impl_min() const {
    return 1;
  }
  [[nodiscard]] virtual size_t impl_max() const {
    return std::numeric_limits<size_t>::max();
  };
  virtual void impl_inject_dependency(Backend* dependency) {}
};

/**
 * @brief A specialized backend that supports only single input and no
 * dependency.
 *
 * Enforces the constraint that only one input is allowed and no dependency can
 * be provided.
 */
class OMNI_EXPORT BackendOne : public Backend {
 private:
  /**
   * @brief Returns the maximum number of inputs supported (always 1).
   *
   * @return Maximum number of inputs supported.
   */
  [[nodiscard]] size_t impl_max() const override final {
    return 1;
  }

  /**
   * @brief Processes a single input/output.
   *
   * Must be implemented by derived classes.
   *
   * @param io Single input/output data to be processed.
   */
  virtual void forward(const dict& io) {
    forward_with_dep(io, *this);
  }

  virtual void forward_with_dep(const dict& io, Backend& dep) {
    throw std::runtime_error("Not Implemented");
  }

  /**
   * @brief Processes input/output data with single input.
   *
   * @param io Input/output data to be processed.
   * @throws std::invalid_argument If more than one input is provided.
   */
  void impl_forward(const std::vector<dict>& io) override final;

  void impl_forward_with_dep(const std::vector<dict>& ios, Backend& dep)
      override final;
};

/**
 * @brief A specialized backend that supports only single input and no
 * dependency.
 *
 * Enforces the constraint that only one input is allowed and no dependency can
 * be provided.
 */
class OMNI_EXPORT BackendMax : public Backend {
 private:
  [[nodiscard]] size_t impl_max() const override final {
    return std::numeric_limits<std::size_t>::max();
  }
  void impl_forward(const std::vector<dict>& io) override;
};

/**
 * @brief Base class for all backends that support event-driven processing.
 */
class HasEventForwardGuard : public Backend {
 private:
  void impl_forward(const std::vector<dict>& inputs) override final;
  /**
   * @brief Processes input/output data with an event. inputs[i] has key
   * TASK_EVENT_KEY.
   * @reference see Event
   */
  virtual void evented_forward(const std::vector<dict>& inputs) = 0;
};

/**
 * @brief Create a backend instance and register it with the given
 * registered_name. This allows the instance to be retrieved later by the
 * registered name. This function is placed here mainly to solve the issue of
 * static variables not being able to cross dynamic libraries during reflection.
 */
OMNI_EXPORT std::unique_ptr<Backend> create_backend(
    const std::string& class_name,
    const std::string& registered_name = "");

OMNI_EXPORT void register_backend(
    const std::string& aspect_name_str,
    std::unique_ptr<Backend>&& backend);
OMNI_EXPORT void register_backend(
    const std::string& aspect_name_str,
    std::shared_ptr<Backend> backend);
OMNI_EXPORT Backend* get_backend(const std::string& aspect_name_str);
OMNI_EXPORT void unregister_backend(const std::string& aspect_name_str);
OMNI_EXPORT void clearup_backend();
OMNI_EXPORT std::unique_ptr<Backend> init_backend(
    const std::string& backend,
    std::unordered_map<std::string, std::string> config,
    const dict& kwargs = nullptr,
    const std::string& registered_name = "");

#define BACKEND_CLASS(ClassName)                                          \
  class ClassName : public omniback::Backend {                            \
   public:                                                                \
    void impl_init(                                                       \
        const std::unordered_map<std::string, std::string>& config,       \
        const omniback::dict& kwargs) override;                           \
    void impl_forward(const std::vector<omniback::dict>& input) override; \
  };

namespace backend {
void evented_forward(Backend& self, const std::vector<dict>& inputs);
// void evented_forward(const Backend* self, const std::vector<dict>& inputs);

std::string get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::optional<std::string>& defualt_cls_name = std::nullopt);

} // namespace backend
} // namespace omniback

namespace omniback::parser_v2 {

bool get_backend_name(const Backend* obj_ptr, std::string& cls_name);
std::string get_backend_name(const Backend* obj_ptr);

std::string get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::optional<std::string>& defualt_cls_name = std::nullopt);

std::optional<std::string> get_opt_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config);

using ArgsKwargs = std::pair<
    std::vector<std::string>,
    std::unordered_map<std::string, std::string>>;
ArgsKwargs get_args_kwargs(
    const Backend* obj_ptr,
    std::string cls_name,
    const std::unordered_map<std::string, std::string>& config);
} // namespace omniback::parser_v2