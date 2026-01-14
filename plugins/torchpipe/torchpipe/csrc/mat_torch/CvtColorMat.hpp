
#pragma once
#include <omniback/extension.hpp>

namespace torchpipe {

class CvtColorMat : public om::BackendOne {
 public:
  virtual void impl_init(
      const std::unordered_map<std::string, std::string>&,
      const om::dict&) override;

  virtual void forward(const om::dict&) override;

 private:
  std::string color_;
};
} // namespace torchpipe