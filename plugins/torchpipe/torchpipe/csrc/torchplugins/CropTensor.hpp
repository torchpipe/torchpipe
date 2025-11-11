#pragma once

#include <omniback/extension.hpp>
#include <string>
#include <unordered_set>

using omniback::dict;

namespace torchpipe {
class CropTensor : public omniback::BackendOne {
 public:
  virtual void forward(const dict&) override;

 private:
};
} // namespace torchpipe