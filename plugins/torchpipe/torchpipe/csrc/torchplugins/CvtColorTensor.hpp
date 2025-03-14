#pragma once

#include <unordered_set>
#include <string>
#include <hami/extension.hpp>

using hami::dict;

namespace torchpipe {
class CvtColorTensor : public hami::BackendOne {
   private:
    void impl_init(
        const std::unordered_map<std::string, std::string>& config_param,
        const dict& kwargs) override;
    void forward(const dict& input_output) override;

   private:
    std::string color_;
};
}  // namespace torchpipe