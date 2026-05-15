#include <tvm/ffi/c_api.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace orchid {

using namespace tvm::ffi;

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

Tensor EmptyCPUInt32(int64_t n) {
  DLDevice cpu_device{kDLCPU, 0};
  DLDataType int32_dtype{kDLInt, 32, 1};
  return tvm::ffi::Tensor::FromNDAlloc(CPUNDAlloc(), tvm::ffi::ShapeView({n}), int32_dtype, cpu_device);
}

Tensor EmptyCPUInt64(int64_t n) {
  DLDevice cpu_device{kDLCPU, 0};
  DLDataType int64_dtype{kDLInt, 64, 1};
  return tvm::ffi::Tensor::FromNDAlloc(CPUNDAlloc(), tvm::ffi::ShapeView({n}), int64_dtype, cpu_device);
}

Tensor PackPaddedPrefixInt64(Tensor padded_tokens, Tensor lengths) {
  const int n = int(padded_tokens.size(0));
  const int w = int(padded_tokens.size(1));
  const int32_t* p_len = static_cast<const int32_t*>(lengths.data_ptr());
  const int64_t* src = static_cast<const int64_t*>(padded_tokens.data_ptr());

  int64_t total = 0;
  for (int i = 0; i < n; ++i) {
    const int32_t len = std::max<int32_t>(0, p_len[i]);
    total += int64_t(len);
  }

  Tensor out = EmptyCPUInt64(total);
  int64_t* o = static_cast<int64_t*>(out.data_ptr());
  int64_t off = 0;
  for (int i = 0; i < n; ++i) {
    const int32_t len = std::max<int32_t>(0, p_len[i]);
    const int64_t* row = src + int64_t(i) * int64_t(w);
    for (int j = 0; j < len; ++j) {
      o[off + j] = row[j];
    }
    off += int64_t(len);
  }
  return out;
}

Array<Tensor> ScheduleStepPlan(
    Tensor running_req_ids,
    Tensor is_prefill_flags,
    Tensor prefill_remaining,
    Tensor decode_ready,
    int max_batch_tokens,
    int prefill_max_chunk,
    double prefill_budget_fraction,
    int decode_reserve_reqs,
    int strict_no_mix) {
  const int n = int(running_req_ids.numel());
  const int32_t* p_req_ids = static_cast<const int32_t*>(running_req_ids.data_ptr());
  const int32_t* p_is_prefill = static_cast<const int32_t*>(is_prefill_flags.data_ptr());
  const int32_t* p_prefill_rem = static_cast<const int32_t*>(prefill_remaining.data_ptr());
  const int32_t* p_decode_ready = static_cast<const int32_t*>(decode_ready.data_ptr());

  const int cap_tokens = int(max_batch_tokens);
  const int prefill_cap = std::max(1, int(prefill_max_chunk));
  double frac = prefill_budget_fraction;
  if (frac < 0.0) frac = 0.0;
  if (frac > 1.0) frac = 1.0;
  const int prefill_budget_limit = int(double(cap_tokens) * frac);
  const int reserve_decode = std::max(0, int(decode_reserve_reqs));
  const bool no_mix = (int(strict_no_mix) != 0);

  int decode_candidates = 0;
  for (int i = 0; i < n; ++i) {
    if (p_is_prefill[i] == 0 && p_decode_ready[i] != 0) decode_candidates += 1;
  }
  const int reserve = std::min(reserve_decode, decode_candidates);

  std::vector<int32_t> sel_ids;
  std::vector<int32_t> sel_new_tokens;
  std::vector<int32_t> sel_is_prefill;
  std::vector<int32_t> sel_should_output;
  sel_ids.reserve(size_t(n));
  sel_new_tokens.reserve(size_t(n));
  sel_is_prefill.reserve(size_t(n));
  sel_should_output.reserve(size_t(n));

  int used_tokens = 0;
  int used_prefill_tokens = 0;
  bool has_prefill = false;

  for (int i = 0; i < n; ++i) {
    const int32_t req_id = p_req_ids[i];
    const bool is_prefill = (p_is_prefill[i] != 0);

    if (is_prefill) {
      const int rem = int(p_prefill_rem[i]);
      if (rem <= 0) continue;
      int budget_left = cap_tokens - used_tokens;
      if (budget_left <= 0) continue;
      if (!no_mix) {
        budget_left = std::max(0, budget_left - reserve);
        int prefill_budget_left = prefill_budget_limit - used_prefill_tokens;
        budget_left = std::min(budget_left, std::max(0, prefill_budget_left));
      }
      if (no_mix && rem > budget_left) continue;
      int chunk = rem;
      if (!no_mix) chunk = std::min(chunk, budget_left);
      if (!no_mix) chunk = std::min(chunk, prefill_cap);
      if (chunk <= 0) continue;
      if (used_tokens + chunk > cap_tokens) continue;
      sel_ids.push_back(req_id);
      sel_new_tokens.push_back(int32_t(chunk));
      sel_is_prefill.push_back(1);
      sel_should_output.push_back(int32_t(no_mix ? 1 : (chunk >= rem ? 1 : 0)));
      used_tokens += chunk;
      used_prefill_tokens += chunk;
      has_prefill = true;
      continue;
    }

    if (no_mix && has_prefill) continue;
    if (p_decode_ready[i] == 0) continue;
    if (used_tokens + 1 > cap_tokens) continue;
    sel_ids.push_back(req_id);
    sel_new_tokens.push_back(1);
    sel_is_prefill.push_back(0);
    sel_should_output.push_back(1);
    used_tokens += 1;
  }

  Tensor out_ids = EmptyCPUInt32(int64_t(sel_ids.size()));
  Tensor out_new_tokens = EmptyCPUInt32(int64_t(sel_new_tokens.size()));
  Tensor out_is_prefill = EmptyCPUInt32(int64_t(sel_is_prefill.size()));
  Tensor out_should_output = EmptyCPUInt32(int64_t(sel_should_output.size()));

  int32_t* o_ids = static_cast<int32_t*>(out_ids.data_ptr());
  int32_t* o_nt = static_cast<int32_t*>(out_new_tokens.data_ptr());
  int32_t* o_pf = static_cast<int32_t*>(out_is_prefill.data_ptr());
  int32_t* o_so = static_cast<int32_t*>(out_should_output.data_ptr());
  for (size_t i = 0; i < sel_ids.size(); ++i) {
    o_ids[i] = sel_ids[i];
    o_nt[i] = sel_new_tokens[i];
    o_pf[i] = sel_is_prefill[i];
    o_so[i] = sel_should_output[i];
  }

  Array<Tensor> ret;
  ret.reserve(4);
  ret.push_back(std::move(out_ids));
  ret.push_back(std::move(out_new_tokens));
  ret.push_back(std::move(out_is_prefill));
  ret.push_back(std::move(out_should_output));
  return ret;
}

}  // namespace orchid

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("custom.schedule_step_plan", orchid::ScheduleStepPlan);
  refl::GlobalDef().def("custom.pack_padded_prefix_i64", orchid::PackPaddedPrefixInt64);
}
