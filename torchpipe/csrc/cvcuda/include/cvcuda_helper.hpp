#pragma once
#include <torch/torch.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Allocator.hpp>

namespace ipipe {
nvcv::Tensor toNvcvTensor(const torch::Tensor& src, std::string data_format = "");
torch::Tensor fromNvcvTensor(const nvcv::Tensor& src);

nvcv::Allocator nvcv_torch_allocator();
}  // namespace ipipe
