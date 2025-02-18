#pragma once

#include <unordered_set>
#include <string>
#include <hami/extension.hpp>

using hami::dict;

namespace torchpipe {
class CropTensor : public hami::SingleBackend {
 public:
  virtual void forward(const dict&) override;

 private:
};
}  // namespace torchpipe