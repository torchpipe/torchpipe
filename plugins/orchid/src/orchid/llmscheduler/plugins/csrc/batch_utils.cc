#include "batch_utils.h"
#include <iostream>

namespace custom {

struct CPUNDAlloc {
    void AllocData(DLTensor* tensor) {
        size_t data_size = tvm::ffi::GetDataSize(*tensor);
        tensor->data = malloc(data_size);
    }
    void FreeData(DLTensor* tensor) {
        if (tensor->data != nullptr) {
            free(tensor->data);
            tensor->data = nullptr;
        }
    }
};

BatchMetadataBuilder::BatchMetadataBuilder(std::shared_ptr<PageManager> pm) : pm_(pm) {}

BatchMetadata BatchMetadataBuilder::PrepareStep(
    tvm::ffi::TensorView req_ids,
    tvm::ffi::TensorView total_lens,
    tvm::ffi::TensorView new_tokens,
    int page_size,
    int layer_idx,
    int num_layers) 
{
    BatchMetadata meta;
    int batch_size = req_ids.numel();

    const int32_t* p_req_ids = static_cast<const int32_t*>(req_ids.data_ptr());
    const int32_t* p_total_lens = static_cast<const int32_t*>(total_lens.data_ptr());
    const int32_t* p_new_tokens = static_cast<const int32_t*>(new_tokens.data_ptr());

    // Pre-calculate sizes first to avoid reallocations
    int64_t total_indices = 0;
    int64_t total_slots = 0;
    
    for (int i = 0; i < batch_size; ++i) {
        int total_len = p_total_lens[i];
        int new_len = p_new_tokens[i];
        total_indices += (total_len + page_size - 1) / page_size;
        total_slots += new_len;
    }

    DLDevice cpu_device{kDLCPU, 0};
    DLDataType int32_dtype{kDLInt, 32, 1};

    // Pre-allocate tensors first
    meta.indptr = tvm::ffi::Tensor::FromNDAlloc(
        CPUNDAlloc(),
        tvm::ffi::ShapeView({static_cast<int64_t>(batch_size + 1)}),
        int32_dtype,
        cpu_device
    );
    meta.indices = tvm::ffi::Tensor::FromNDAlloc(
        CPUNDAlloc(),
        tvm::ffi::ShapeView({total_indices}),
        int32_dtype,
        cpu_device
    );
    meta.last_page_len = tvm::ffi::Tensor::FromNDAlloc(
        CPUNDAlloc(),
        tvm::ffi::ShapeView({static_cast<int64_t>(batch_size)}),
        int32_dtype,
        cpu_device
    );
    meta.qo_indptr = tvm::ffi::Tensor::FromNDAlloc(
        CPUNDAlloc(),
        tvm::ffi::ShapeView({static_cast<int64_t>(batch_size + 1)}),
        int32_dtype,
        cpu_device
    );
    meta.slot_mapping = tvm::ffi::Tensor::FromNDAlloc(
        CPUNDAlloc(),
        tvm::ffi::ShapeView({total_slots}),
        int32_dtype,
        cpu_device
    );
    meta.batch_indices = tvm::ffi::Tensor::FromNDAlloc(
        CPUNDAlloc(),
        tvm::ffi::ShapeView({total_slots}),
        int32_dtype,
        cpu_device
    );
    meta.positions = tvm::ffi::Tensor::FromNDAlloc(
        CPUNDAlloc(),
        tvm::ffi::ShapeView({total_slots}),
        int32_dtype,
        cpu_device
    );

    // Get raw pointers to tensor data
    int32_t* p_indptr = static_cast<int32_t*>(meta.indptr.data_ptr());
    int32_t* p_indices = static_cast<int32_t*>(meta.indices.data_ptr());
    int32_t* p_last_page_len = static_cast<int32_t*>(meta.last_page_len.data_ptr());
    int32_t* p_qo_indptr = static_cast<int32_t*>(meta.qo_indptr.data_ptr());
    int32_t* p_slot_mapping = static_cast<int32_t*>(meta.slot_mapping.data_ptr());
    int32_t* p_batch_indices = static_cast<int32_t*>(meta.batch_indices.data_ptr());
    int32_t* p_positions = static_cast<int32_t*>(meta.positions.data_ptr());

    // Fill indptr[0] and qo_indptr[0]
    p_indptr[0] = 0;
    p_qo_indptr[0] = 0;

    int current_page_offset = 0;
    int current_qo_offset = 0;
    int slot_idx = 0;

    for (int i = 0; i < batch_size; ++i) {
        int req_id = p_req_ids[i];
        int total_len = p_total_lens[i];
        int new_len = p_new_tokens[i];

        int unique_id = req_id;
        int num_pages = (total_len + page_size - 1) / page_size;
        std::vector<int32_t> pages = pm_->GetPages(unique_id, num_pages);

        // Copy pages directly to indices
        memcpy(p_indices + current_page_offset, pages.data(), num_pages * sizeof(int32_t));
        current_page_offset += num_pages;
        p_indptr[i + 1] = current_page_offset;

        int last_len = (total_len - 1) % page_size + 1;
        p_last_page_len[i] = last_len;

        current_qo_offset += new_len;
        p_qo_indptr[i + 1] = current_qo_offset;

        int start_pos = total_len - new_len;
        for (int t = 0; t < new_len; ++t) {
            int abs_pos = start_pos + t;
            int page_idx = abs_pos / page_size;
            int page_offset = abs_pos % page_size;

            if (page_idx >= pages.size()) {
                 std::cerr << "Error: page_idx out of bounds" << std::endl;
                 continue;
            }

            int physical_page = pages[page_idx];
            int slot = physical_page * page_size + page_offset;
            p_slot_mapping[slot_idx++] = slot;
            p_batch_indices[slot_idx - 1] = int32_t(i);
            p_positions[slot_idx - 1] = int32_t(abs_pos);
        }
    }

    return meta;
}

} // namespace custom
