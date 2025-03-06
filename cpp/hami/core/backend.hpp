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
#include <unordered_map>
#include <string>
#include <vector>
#include <limits>
#include <memory>
#include <stdexcept>

#include "hami/core/dict.hpp"
#include "hami/helper/hami_export.h"
#include "hami/core/reflect.h"

namespace hami {

/**
 * @brief Base class for all backends.
 *
 * Provides the interface for initializing and processing data.
 */
class HAMI_EXPORT Backend {
   public:
    virtual ~Backend() = default;  ///< Default virtual destructor.

    /**
     * @brief Initializes the backend with configuration and shared data.
     *
     * Must be called before any other operations. After successful
     * initialization, the values of `max()` and `min()` will not change.
     *
     * @param config Configuration parameters for the backend.
     * @param dict_config Shared data accessible across multiple backends.
     * @throws any Exception thrown during initialization.
     */
    virtual void init(const std::unordered_map<string, string>& config,
                      const dict& dict_config) {}

    /**
     * @brief Processes input/output data.
     *
     * The size of `input_output` must be within the range [min(), max()].
     *
     * @param input_output Input/output data to be processed.
     */
    virtual void forward(const std::vector<dict>& input_output) {}

    /**
     * @brief Returns the maximum number of inputs supported.
     *
     * This value is determined after `init()` and `inject_dependency` are
     * called.
     *
     * @return Maximum number of inputs supported. Default to
     * std::numeric_limits<size_t>::max()
     */
    [[nodiscard]] virtual size_t max() const {
        return std::numeric_limits<size_t>::max();
    }

    /**
     * @brief Returns the minimum number of inputs supported.
     *
     * This value is determined after `init()` and `inject_dependency` are
     * called.
     *
     * @return Minimum number of inputs supported.
     */
    [[nodiscard]] virtual size_t min() const { return 1; }

    /**
     * @brief Sets the dependency for the backend.
     *
     * This method allows setting a Backend object as a dependency for the
     * current backend. It should be called before the normal forward method if
     * a dependency is required. Backend implementers can use this dependency to
     * perform actual work. This is different from the dependency in the forward
     * method, which is a runtime dependency.
     *
     * This is primarily used for Inversion of Control (IoC) or Dependency
     * Injection (DI) in the scheduling system. For typical computation-type
     * backends, this can be ignored.
     *
     * @param dependency Pointer to the backend dependency.
     * @throws std::runtime_error If dependency setting is not supported by this
     * backend.
     */
    virtual void inject_dependency(Backend* dependency) {
        throw std::runtime_error("inject_dependency not supported by default");
    }

    /**
     * @brief Processes input/output data with a runtime dependency.
     *
     * @param input_output Input/output data to be processed.
     * @param dependency Pointer to the backend dependency.
     */
    virtual void forward(const std::vector<dict>& input_output,
                         Backend* dependency) {
        if (dependency)
            throw std::runtime_error(
                "forward(input_output, dependency) not supported by default");
        else
            forward(input_output);
    }

    Backend() = default;
    Backend(const Backend&) = delete;
    Backend(const Backend&&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend& operator=(const Backend&&) = delete;

    /**
     * @brief Helper function for @ref forward to check the range of [min(),
     * max()].
     *
     * This function ensures that the size of `input_output` is within the valid
     * range defined by min() and max(). If the size is within the range, it
     * calls the forward method. If the size exceeds max(), it processes the
     * input in chunks. If the size is less than min(), it throws an
     * invalid_argument exception.
     *
     * @param input_output The input/output data to be processed.
     * @throws std::invalid_argument if the size of input_output is less than
     * min().
     */
    void safe_forward(const std::vector<dict>& input_output);

    /**
     * @brief Helper function to reflect the class name X, if an instance is
     * created via HAMI_CREATE(Backend, X). Here, X is a subclass of Backend.
     * X may have multiple aliases during registration: HAMI_REGISTER(Backend,
     * X, "X,X2"). This function correctly distinguishes whether the instance
     * created via HAMI_CREATE was instantiated with X or X2 as the parameter.
     */
    // void get_class_name(std::string& default_name) const;
};

using MaxBackend = Backend;

/**
 * @brief Base class for all backends that support event-driven processing.
 */
class HasEventForwardGuard : public Backend {
   public:
    virtual void forward(const std::vector<dict>& inputs) override final;
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
HAMI_EXPORT std::unique_ptr<Backend> create_backend(
    const std::string& class_name, const std::string& registered_name = "");

HAMI_EXPORT void register_backend(const std::string& aspect_name_str,
                                  std::unique_ptr<Backend>&& backend);
HAMI_EXPORT void register_backend(const std::string& aspect_name_str,
                                  std::shared_ptr<Backend> backend);
HAMI_EXPORT Backend* get_backend(const std::string& aspect_name_str);
HAMI_EXPORT void unregister_backend(const std::string& aspect_name_str);
HAMI_EXPORT void clearup_backend();
HAMI_EXPORT std::unique_ptr<Backend> init_backend(
    const std::string& backend,
    std::unordered_map<std::string, std::string> config,
    const dict& dict_config = nullptr, const std::string& registered_name = "");

#define BACKEND_CLASS(ClassName)                                              \
    class ClassName : public hami::Backend {                                  \
       public:                                                                \
        void init(const std::unordered_map<std::string, std::string>& config, \
                  const hami::dict& dict_config) override;                    \
        void forward(const std::vector<hami::dict>& input) override;          \
    };

namespace backend {
void evented_forward(Backend& self, const std::vector<dict>& inputs);
// void evented_forward(const Backend* self, const std::vector<dict>& inputs);

/**
 * Checks if either all inputs have events or none have events
 *
 * @param inputs Vector of dictionary items to check
 * @return true if all items have events or if no items have events
 * @return false if there's a mix of items with and without events
 */
bool is_none_or_all_evented_and_unempty(const std::vector<dict>& inputs);
bool is_all_evented(const std::vector<dict>& inputs);

std::string get_dependency_name(
    const Backend* this_ptr,
    const std::unordered_map<std::string, std::string>& config,
    const std::optional<std::string>& defualt_cls_name = std::nullopt);

}  // namespace backend
}  // namespace hami