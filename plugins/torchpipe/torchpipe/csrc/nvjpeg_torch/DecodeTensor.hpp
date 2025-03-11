#pragma once
#include <hami/extension.hpp>
#include "nvjpeg.h"

namespace torchpipe {

class DecodeTensor : public hami::SingleBackend {
   public:
    virtual void impl_init(const std::unordered_map<std::string, std::string>&,
                           const hami::dict&) override;

    virtual void forward(const hami::dict&) override;
    ~DecodeTensor();

   private:
    nvjpegHandle_t handle_;
    nvjpegJpegState_t state_;
    std::string color_{"rgb"};
    std::string data_format_{"nchw"};
    //   std::unique_ptr<Params> params_;
};
}  // namespace torchpipe