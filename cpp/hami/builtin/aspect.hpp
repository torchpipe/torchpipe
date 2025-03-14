#pragma once

#include <string>
#include <vector>
#include "hami/builtin/basic_backends.hpp"

namespace hami {

class Aspect : public Container {
   public:
    virtual void post_init(const std::unordered_map<std::string, std::string>&,
                           const dict&) override final;

    /**
     * @brief  select a sub-backend.
     */
    virtual void impl_forward(const std::vector<dict>&) override;

    void impl_inject_dependency(Backend* dependency) override final {
        base_dependencies_.front()->inject_dependency(dependency);
    }

   private:
    virtual std::pair<size_t, size_t> update_min_max(
        const std::vector<Backend*>& depends) override;

   private:
};

// class DefineBackend : public Backend {
//    public:
//     void impl_init(const std::unordered_map<std::string, std::string>&
//     config,
//               const dict& kwargs) override;

//     void impl_inject_dependency(Backend* dependency) override final {
//         if (!proxy_backend_) {
//             throw std::runtime_error("DefineBackend was not initialized
//             yet");
//         } else
//             proxy_backend_->inject_dependency(dependency);
//     }

//     void impl_forward(const std::vector<dict>& inputs,
//                  Backend* dependency) override {
//         proxy_backend_->forward(inputs, dependency);
//     }

//     void impl_forward(const std::vector<dict>& inputs) override {
//         proxy_backend_->forward(inputs);
//     }
//     [[nodiscard]] virtual size_t impl_max() const override {
//         return proxy_backend_->max();
//     }

//     [[nodiscard]] virtual size_t impl_min() const override {
//         return proxy_backend_->min();
//     }

//    protected:
//     // Backend* dependency_{nullptr};
//     // Backend* proxy_backend_{nullptr};
//     std::unique_ptr<Backend> proxy_backend_;

//    private:
//     virtual std::pair<std::string, str::str_map> order() const;
// };
}  // namespace hami

// #define HAMI_DEFINE_BACKEND(ClsName, main_backend, config)