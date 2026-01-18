
#pragma once
#include <omniback/extension.hpp>

namespace torchpipe {

class DecodeMat : public om::BackendOne {
 public:
  virtual void impl_init(
      const std::unordered_map<std::string, std::string>&,
      const om::dict&) override;

  virtual void forward(const om::dict&) override;

 private:
  std::string color_{"rgb"};
  std::string data_format_{"nchw"};
};
} // namespace torchpipe