
#pragma once
#include <hami/extension.hpp>

namespace torchpipe {

class Mat2Tensor : public hami::SingleBackend {
   public:
    virtual void impl_init(const std::unordered_map<std::string, std::string>&,
                           const hami::dict&) override;

    virtual void forward(const hami::dict&) override;

   private:
    std::string device_{"cuda"};
};

}  // namespace torchpipe