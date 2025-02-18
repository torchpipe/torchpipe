
#pragma once
#include <hami/extension.hpp>

namespace torchpipe {

class DecodeMat : public hami::SingleBackend {
 public:
  virtual void init(const std::unordered_map<std::string, std::string>&,
                    const hami::dict&) override;

  virtual void forward(const hami::dict&) override;

 private:
  std::string color_{"rgb"};
  std::string data_format_{"nchw"};
};
}  // namespace torchpipe