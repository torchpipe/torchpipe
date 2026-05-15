#include "page_manager.h"
#include "batch_utils.h"
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>
#include <vector>
#include <memory>

namespace custom {

using namespace tvm::ffi;

// Global instances
static std::shared_ptr<PageManager> PM;
static std::unique_ptr<BatchMetadataBuilder> BMB;
static int PM_MAX_PAGES = 0;

void Init(int max_pages) {
    const int mp = int(max_pages);
    if (!PM || PM_MAX_PAGES != mp) {
        PM = std::make_shared<PageManager>(mp);
        BMB = std::make_unique<BatchMetadataBuilder>(PM);
        PM_MAX_PAGES = mp;
        return;
    }
    PM->Reset();
}

// Wrapper for existing get_pages functionality (for backward compatibility or direct use)
std::vector<int32_t> GetPagesFunc(int req_id, int num_needed) {
    if (!PM) Init(4096); 
    return PM->GetPages(req_id, num_needed);
}

void FreeFunc(int req_id) {
    if (PM) PM->Free(req_id);
}

void ResetFunc() {
    if (PM) PM->Reset();
}

// Batch Metadata Binding
// Returns an array of Tensors: [indptr, indices, last_page_len, qo_indptr, slot_mapping, batch_indices, positions]
Array<Tensor> PrepareStepFunc(
    Tensor req_ids,
    Tensor total_lens,
    Tensor new_tokens,
    int page_size,
    int layer_idx,
    int num_layers
) {
    if (!BMB) Init(4096);
    
    BatchMetadata meta = BMB->PrepareStep(req_ids, total_lens, new_tokens, page_size, layer_idx, num_layers);
    
    Array<Tensor> ret;
    ret.reserve(7);
    ret.push_back(std::move(meta.indptr));
    ret.push_back(std::move(meta.indices));
    ret.push_back(std::move(meta.last_page_len));
    ret.push_back(std::move(meta.qo_indptr));
    ret.push_back(std::move(meta.slot_mapping));
    ret.push_back(std::move(meta.batch_indices));
    ret.push_back(std::move(meta.positions));
    
    return ret;
}

} // namespace custom

// Register global functions using reflection::GlobalDef
TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
    refl::GlobalDef()
        .def("custom.init", custom::Init)
        .def("custom.get_pages", custom::GetPagesFunc)
        .def("custom.free", custom::FreeFunc)
        .def("custom.reset", custom::ResetFunc)
        .def("custom.prepare_step", custom::PrepareStepFunc);
}
