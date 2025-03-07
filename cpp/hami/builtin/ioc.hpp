

#pragma once

#include <string>
#include <vector>
#include "hami/core/backend.hpp"
namespace hami {

/**
 * @brief Inversion of Control container managing backend initialization and
 * execution phases. Usage: IoC[InitBackends;ExecutionBackends] (e.g.,
 * IoC[A,B[C];D[E,A]])
 *
 * @note
 * - Initialization phase (before semicolon): Initializes backends in
 * declaration order. Created backends are stored in a registry for reuse.
 * - Execution phase (after semicolon): Configurations for runtime operations.
 * Reuses initialized backends from registry when referenced.
 * - Supports compound syntax: (pre=1)BackendName(post=param)[NestedBackend]
 * - All initialized backends share the same dict_config if provided
 */
class IoC : public Backend {
   public:
    void init(const std::unordered_map<std::string, std::string>& config,
              const dict& dict_config);
    void forward(const std::vector<dict>& input_output) override {
        forward_backend_->forward(input_output);
    }
    [[nodiscard]] size_t max() const override {
        return forward_backend_->max();
    }
    [[nodiscard]] size_t min() const override {
        return forward_backend_->min();
    }

    virtual void post_init(
        const std::unordered_map<std::string, std::string>& config,
        const dict& dict_config) {}

    ~IoC() {
        // order is important here
        forward_backend_.release();
        while (!base_dependencies_.empty()) {
            base_dependencies_.pop_back();
        }
    }

   private:
    void init_phase(const std::string& phase_config,
                    const std::unordered_map<std::string, std::string>& config,
                    const dict& dict_config);

   protected:
    // std::unordered_map<std::string, std::unique_ptr<Backend>>
    // backend_registry_; std::vector<std::string> execution_configs_;
    std::vector<std::unique_ptr<Backend>> base_dependencies_;
    std::vector<std::unordered_map<std::string, std::string>> base_config_;
    std::unique_ptr<Backend> forward_backend_;
};

}  // namespace hami