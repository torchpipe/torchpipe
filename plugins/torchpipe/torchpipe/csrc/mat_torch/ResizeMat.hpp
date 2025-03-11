
#pragma once
#include <hami/extension.hpp>

namespace torchpipe {

class ResizeMat : public hami::SingleBackend {
   public:
    virtual void impl_init(const std::unordered_map<std::string, std::string>&,
                           const hami::dict&) override;

    virtual void forward(const hami::dict&) override;

   private:
    size_t resize_h_;
    size_t resize_w_;
};
}  // namespace torchpipe