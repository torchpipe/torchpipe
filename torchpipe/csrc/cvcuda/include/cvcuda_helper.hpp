#pragma once
#include <ATen/ATen.h>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/Allocator.hpp>

namespace ipipe {
nvcv::Tensor toNvcvTensor(const at::Tensor& src, std::string data_format = "");
at::Tensor fromNvcvTensor(const nvcv::Tensor& src);

nvcv::Allocator nvcv_torch_allocator();
}  // namespace ipipe
