#pragma once
#include "page_manager.h"
#include <vector>
#include <cstdint>
#include <memory>
#include <tvm/ffi/container/tensor.h>

namespace custom {

struct BatchMetadata {
    tvm::ffi::Tensor indptr;
    tvm::ffi::Tensor indices;
    tvm::ffi::Tensor last_page_len;
    tvm::ffi::Tensor qo_indptr;
    tvm::ffi::Tensor slot_mapping;
    tvm::ffi::Tensor batch_indices;
    tvm::ffi::Tensor positions;
};

class BatchMetadataBuilder {
public:
    explicit BatchMetadataBuilder(std::shared_ptr<PageManager> pm);
    
    // Updates internal state and returns metadata for current step
    BatchMetadata PrepareStep(
        tvm::ffi::TensorView req_ids, 
        tvm::ffi::TensorView total_lens,
        tvm::ffi::TensorView new_tokens,
        int page_size,
        int layer_idx,
        int num_layers
    );

private:
    std::shared_ptr<PageManager> pm_;
};

} // namespace custom
