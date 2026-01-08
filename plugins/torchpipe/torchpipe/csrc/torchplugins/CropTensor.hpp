#pragma once

#include <omniback/extension.hpp>
#include <string>
#include <unordered_set>

using om::dict;

namespace torchpipe {
class CropTensor : public om::BackendOne {
 public:
  virtual void forward(const dict&) override;

 private:
};

class CopyMakeBorderTensor : public om::BackendOne {
 public:
  virtual void forward(const dict&) override;

 private:
  int top_{0};
  int bottom_{0};
  int left_{0};
  int right_{0};
};

// class WarpAffineTensor : public om::BackendOne {
//  public:
//   virtual void impl_init(
//       const std::unordered_map<std::string, std::string>& config,
//       const dict& kwargs) override;
//   virtual void forward(const dict&) override;

//  private:
//   int target_h_{0};
//   int target_w_{0};
// };

} // namespace torchpipe