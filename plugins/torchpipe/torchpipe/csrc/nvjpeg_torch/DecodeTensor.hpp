#pragma once
#include <omniback/extension.hpp>
#include <omniback/addons/torch/type_traits.h>
#include "nvjpeg.h"

namespace torchpipe {

class DecodeTensor : public omniback::BackendOne {
 public:
  virtual void impl_init(
      const std::unordered_map<std::string, std::string>&,
      const omniback::dict&) override;

  virtual void forward(const omniback::dict&) override;
  ~DecodeTensor();

 private:
  nvjpegHandle_t handle_;
  nvjpegJpegState_t state_;
  std::string color_{"rgb"};
  std::string data_format_{"nchw"};
  //   std::unique_ptr<Params> params_;
};
} // namespace torchpipe