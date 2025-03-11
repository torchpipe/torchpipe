#pragma once

#include <memory>
#include "hami/builtin/basic_backends.hpp"
#include "hami/helper/params.hpp"
#include "hami/helper/macro.h"
namespace hami {

class ResultParser : public Dependency {
   private:
    std::function<void(const dict&)> parser_;

   public:
    void pre_init(const std::unordered_map<std::string, std::string>& config,
                  const dict& dict_config) override final;
    virtual void init_dep_impl(
        const std::unordered_map<std::string, std::string>& config,
        const dict& dict_config) {}
    void custom_forward_with_dep(const std::vector<dict>& inputs,
                                 Backend* dependency) override final {
        dependency->safe_forward(inputs);
        for (const auto& item : inputs) {
            parser_(item);
        }
    }

    virtual std::function<void(const dict&)> parser_impl() const = 0;
};

}  // namespace hami