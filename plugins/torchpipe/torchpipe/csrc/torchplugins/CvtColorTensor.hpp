#pragma once

#include <omniback/extension.hpp>
#include <string>
#include <unordered_set>

using omniback::dict;

namespace torchpipe {
class CvtColorTensor : public omniback::BackendOne {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config_param,
      const dict& kwargs) override;
  void forward(const dict& input_output) override;

 private:
  std::string color_;
};

} // namespace torchpipe