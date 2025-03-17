#pragma once

#include <string>
#include <vector>
#include "hami/core/backend.hpp"

namespace hami {
class ControlPlane : public Backend {
   private:
    void impl_init(const std::unordered_map<std::string, std::string> &params,
                   const dict &options) override final;

    virtual void impl_custom_init(
        const std::unordered_map<std::string, std::string> &params,
        const dict &options) = 0;

    // Default class name if the instance is not create via reflection.
    virtual std::string default_cls_name() const { return "ControlPlane"; }

   protected:
    std::vector<std::pair<std::vector<std::string>,
                          std::unordered_map<std::string, std::string>>>
        prefix_args_kwargs_;
    std::vector<std::string> backend_cfgs_;
};

}  // namespace hami