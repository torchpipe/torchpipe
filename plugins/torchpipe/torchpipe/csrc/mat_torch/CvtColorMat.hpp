
#pragma once
#include <omniback/extension.hpp>

namespace torchpipe {

class CvtColorMat : public omniback::BackendOne {
 public:
  virtual void impl_init(
      const std::unordered_map<std::string, std::string>&,
      const omniback::dict&) override;

  virtual void forward(const omniback::dict&) override;

 private:
  std::string color_;
};
} // namespace torchpipe